from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.analysis.research_report import (
    build_research_analysis,
    exact_mcnemar,
    wilson_interval,
)
from embodied_gap.core.task_schema import Task
from embodied_gap.core.task_schema import dump_jsonl
from embodied_gap.core.task_schema import load_tasks
from embodied_gap.datasets.eai_adapter import (
    EAIAdapterError,
    EmbodiedAgentInterfaceAdapter,
    format_plan_step,
    parse_pddl_problem,
)
from embodied_gap.datasets.taskset_builder import TaskSetBuilder, classify_difficulty
from embodied_gap.datasets.resource_paths import resolve_domain_path, resolve_problem_path
from embodied_gap.datasets.split_freezer import freeze_heldout_split
from embodied_gap.experiments.model_matrix import ModelMatrixConfig, ModelSpec, MultiModelExperimentRunner
from embodied_gap.experiments.pilot_budget import inspect_model_matrix
from embodied_gap.knowledge.corpus_builder import KnowledgeCorpusBuilder
from embodied_gap.knowledge.failure_memory import classify_failure_patterns
from embodied_gap.knowledge.failure_memory_store import (
    FailureMemoryEntry,
    FrozenFailureMemory,
    build_frozen_failure_memory,
)
from embodied_gap.knowledge.pddl_grounded_search import PDDLGroundedSearch
from embodied_gap.knowledge.retriever import ExampleRetriever
from embodied_gap.evaluation.metrics import evaluate_run
from embodied_gap.experiments.config import ExperimentConfig
from embodied_gap.experiments.runner import ExperimentRunner
from embodied_gap.evaluation.pddl_gold_validator import PDDLGoldPlanValidator
from embodied_gap.evaluation.official_eai import (
    inspect_official_response_tree,
    validate_action_sequencing_records,
)
from embodied_gap.execution.symbolic_executor import SymbolicExecutor
from embodied_gap.harness.controller import HarnessController
from embodied_gap.harness.recovery_policy import HarnessMode
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.clients import OneAPIChatClient
from embodied_gap.llm.prompts import render_planning_prompt
from embodied_gap.planners.graph_grounded import GraphGroundedPlanner
from embodied_gap.planners.prompt_only import (
    EngineeredPromptPlanner,
    MinimalPromptPlanner,
    PromptOnlyPlanner,
)
from embodied_gap.planners.retrieval_augmented import RetrievalAugmentedPlanner


class ResearchFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tasks = load_tasks("data/sample_tasks.jsonl")
        cls.examples = [task for task in tasks if task.split == "train"]
        cls.eval_tasks = {task.id: task for task in tasks if task.split != "train"}

    def test_open_loop_prompt_exposes_missing_step(self) -> None:
        task = self.eval_tasks["eval_move_apple"]
        run = HarnessController().run(task, PromptOnlyPlanner(), HarnessMode.H0_OPEN_LOOP)
        record = evaluate_run(task, run)
        self.assertFalse(record.execution_success)
        self.assertIn("missing_step", record.error_counts)

    def test_prompt_ablation_profiles_are_distinct(self) -> None:
        task = self.eval_tasks["eval_move_apple"]
        minimal = MinimalPromptPlanner().plan(task)
        structured = PromptOnlyPlanner().plan(task)
        engineered = EngineeredPromptPlanner().plan(task)

        self.assertEqual(minimal.planner_name, "B0_minimal_prompt")
        self.assertEqual(structured.planner_name, "P0_structured_prompt")
        self.assertEqual(engineered.planner_name, "P0_engineered_prompt")
        self.assertNotIn("Initial facts:", minimal.prompt)
        self.assertIn("Initial facts:", structured.prompt)
        self.assertIn("Goal facts:", structured.prompt)
        self.assertIn("internally verify", engineered.prompt)

    def test_retrieval_planner_adapts_successful_plan(self) -> None:
        task = self.eval_tasks["eval_move_apple"]
        run = HarnessController().run(
            task,
            RetrievalAugmentedPlanner(self.examples),
            HarnessMode.H0_OPEN_LOOP,
        )
        record = evaluate_run(task, run)
        self.assertTrue(record.task_success)
        self.assertEqual(run.final_plan.metadata["retrieved"], "train_move_mug")

    def test_rag_retrievers_are_normalized_deterministic_and_top_k(self) -> None:
        examples = load_tasks("data/processed/tasksets/rag_train.jsonl")
        query = load_tasks("data/processed/tasksets/balanced_eval_20.jsonl")[0]
        for method in ("lexical", "bm25", "structured"):
            retriever = ExampleRetriever(
                examples,
                method=method,
                field_profile="instruction_state_goal_schema",
            )
            first = retriever.retrieve(query, k=5)
            second = retriever.retrieve(query, k=5)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertTrue(all(0.0 <= item.score <= 1.0 for item in first))
            self.assertEqual(len({item.task.id for item in first}), 5)

        planner = RetrievalAugmentedPlanner(
            self.examples,
            top_k=2,
            retrieval_method="structured",
            field_profile="instruction_goal",
        )
        plan = planner.plan(self.eval_tasks["eval_move_apple"])
        self.assertEqual(len(plan.metadata["retrieved_ids"]), 2)
        self.assertEqual(plan.metadata["retrieval_top_k"], 2)
        self.assertEqual(plan.metadata["retrieval_method"], "structured")
        self.assertIn("Retrieved demonstration 2:", plan.prompt)

    def test_official_virtualhome_contract_requires_name_id_pairs(self) -> None:
        invalid = validate_action_sequencing_records(
            [
                {
                    "identifier": "11_1",
                    "llm_output": '[{"WALK":["floor_lamp"]}]',
                }
            ],
            dataset="virtualhome",
        )
        self.assertFalse(invalid["valid"])
        self.assertEqual(
            invalid["issues"][0]["code"],
            "virtualhome_name_id_pairs_required",
        )

        valid = validate_action_sequencing_records(
            [
                {
                    "identifier": "11_1",
                    "llm_output": '[{"WALK":["floor_lamp",1000]}]',
                }
            ],
            dataset="virtualhome",
        )
        self.assertTrue(valid["valid"])

    def test_official_preflight_does_not_promote_partial_output_to_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            action_dir = Path(temp_dir) / "virtualhome" / "action_sequencing"
            action_dir.mkdir(parents=True)
            (action_dir / "unit_outputs.json").write_text(
                json.dumps(
                    [
                        {
                            "identifier": "11_1",
                            "llm_output": '[{"WALK":["floor_lamp",1000]}]',
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = inspect_official_response_tree(temp_dir)

        self.assertEqual(report["required_slot_count"], 8)
        self.assertEqual(report["present_slot_count"], 1)
        self.assertTrue(report["action_sequencing_shapes_valid"])
        self.assertFalse(report["submission_ready"])

    def test_pilot_preflight_counts_calls_and_excludes_heldout(self) -> None:
        prompt_report = inspect_model_matrix(
            "configs/experiments/pilot_prompt_deepseek_20.json"
        )
        self.assertEqual(prompt_report["task_count"], 20)
        self.assertEqual(prompt_report["run_record_count"], 60)
        self.assertEqual(prompt_report["worst_case_total_llm_call_count"], 60)
        self.assertTrue(prompt_report["safe_for_development_selection"])

        rag_report = inspect_model_matrix(
            "configs/experiments/pilot_rag_bm25_deepseek_20.json"
        )
        self.assertEqual(rag_report["run_record_count"], 20)
        self.assertEqual(rag_report["worst_case_total_llm_call_count"], 20)
        self.assertFalse(rag_report["uses_frozen_heldout"])

    def test_graph_grounded_planner_solves_preconditions(self) -> None:
        task = self.eval_tasks["eval_clean_plate"]
        run = HarnessController().run(task, GraphGroundedPlanner(), HarnessMode.H0_OPEN_LOOP)
        record = evaluate_run(task, run)
        self.assertTrue(record.task_success)
        self.assertFalse(record.risk)
        self.assertIn("search_seconds", run.initial_plan.metadata)

    def test_full_harness_rejects_hazard(self) -> None:
        task = self.eval_tasks["eval_heat_phone_hazard"]
        run = HarnessController().run(task, PromptOnlyPlanner(), HarnessMode.H2_FULL_RECOVERY)
        record = evaluate_run(task, run)
        self.assertEqual(run.final_plan.actions, ("reject()",))
        self.assertTrue(record.safe_success)
        self.assertFalse(record.risk)

    def test_llm_reflection_uses_explicit_feedback_without_pddl_fallback(self) -> None:
        class RepairClient:
            provider = "unit"
            model = "repair-model"

            def generate(self, prompt: str) -> str:
                self.prompt = prompt
                return json.dumps(
                    [
                        "navigate(fridge)",
                        "open(fridge)",
                        "pickup(apple)",
                        "navigate(countertop)",
                        "put(apple, countertop)",
                    ]
                )

        task = self.eval_tasks["eval_move_apple"]
        client = RepairClient()
        planner = PromptOnlyPlanner(llm_client=client)
        initial = PlanCandidate(
            planner_name=planner.name,
            actions=("navigate(fridge)", "pickup(apple)"),
        )
        run = HarnessController(max_retries=1).run(
            task,
            planner,
            HarnessMode.H2_LLM_REFLECTION,
            initial_plan=initial,
        )

        record = evaluate_run(task, run)
        self.assertTrue(record.task_success)
        self.assertFalse(record.initial_task_success)
        self.assertTrue(record.recovered)
        self.assertEqual(run.patches[0].source, "llm_feedback_replan")
        self.assertIn("missing_preconditions", client.prompt)
        self.assertNotIn("Error-specific repair guidance:", client.prompt)
        self.assertNotIn("Frozen failure memory:", client.prompt)
        self.assertNotEqual(run.patches[0].source, "symbolic_replan")

    def test_recovery_modes_are_separate(self) -> None:
        task = self.eval_tasks["eval_move_apple"]
        planner = PromptOnlyPlanner()
        initial = PlanCandidate(
            planner_name=planner.name,
            actions=("navigate(fridge)", "pickup(apple)"),
        )
        local = HarnessController(max_retries=1).run(
            task,
            planner,
            HarnessMode.H2_LOCAL_RECOVERY,
            initial_plan=initial,
        )
        pddl = HarnessController(max_retries=1).run(
            task,
            planner,
            HarnessMode.H2_PDDL_RECOVERY,
            initial_plan=initial,
        )

        self.assertEqual(local.patches[0].source, "local_patch_repair")
        self.assertEqual(pddl.patches[0].source, "symbolic_replan")
        self.assertIn("search_seconds", pddl.patches[0].metadata)

    def test_error_specific_and_frozen_memory_repair_prompts_are_distinct(self) -> None:
        class RepairClient:
            provider = "unit"
            model = "repair-model"

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return json.dumps(
                    [
                        "navigate(fridge)",
                        "open(fridge)",
                        "pickup(apple)",
                        "navigate(countertop)",
                        "put(apple, countertop)",
                    ]
                )

        task = self.eval_tasks["eval_move_apple"]
        memory = FrozenFailureMemory(
            (
                FailureMemoryEntry(
                    id="memory-1",
                    source_task_id="train_move_mug",
                    instruction="Move a mug from a cabinet to a table.",
                    dataset="unit",
                    task_family="move",
                    tags=("move",),
                    error_type="missing_step",
                    failed_plan=("pickup(mug)",),
                    repaired_plan=("open(cabinet)", "pickup(mug)"),
                    repair_source="unit",
                ),
            ),
            sha256="frozen-unit-hash",
        )
        initial = PlanCandidate(
            planner_name="P0_structured_prompt",
            actions=("navigate(fridge)", "pickup(apple)"),
        )

        error_client = RepairClient()
        error_planner = PromptOnlyPlanner(llm_client=error_client)
        error_run = HarnessController(max_retries=1).run(
            task,
            error_planner,
            HarnessMode.H2_ERROR_SPECIFIC,
            initial_plan=initial,
        )
        self.assertTrue(evaluate_run(task, error_run).task_success)
        self.assertIn("Insert actions that establish", error_client.prompts[0])
        self.assertNotIn("Frozen failure memory:", error_client.prompts[0])

        memory_client = RepairClient()
        memory_planner = PromptOnlyPlanner(llm_client=memory_client)
        memory_run = HarnessController(failure_memory=memory, max_retries=1).run(
            task,
            memory_planner,
            HarnessMode.H2_MEMORY,
            initial_plan=initial,
        )
        self.assertTrue(evaluate_run(task, memory_run).task_success)
        self.assertIn("Memory ID: memory-1", memory_client.prompts[0])
        self.assertNotIn("Error-specific repair guidance:", memory_client.prompts[0])
        self.assertEqual(
            memory_run.patches[0].metadata["failure_memory_sha256"],
            "frozen-unit-hash",
        )

    def test_recovery_pilot_budget_keeps_mechanisms_separate(self) -> None:
        isolated = inspect_model_matrix(
            "configs/experiments/pilot_recovery_deepseek_20.json"
        )
        self.assertEqual(isolated["run_record_count"], 120)
        self.assertEqual(isolated["initial_llm_call_count"], 20)
        self.assertEqual(isolated["worst_case_repair_llm_call_count"], 60)
        self.assertNotIn("H2_full_recovery", isolated["harness_modes"])

        combined = inspect_model_matrix(
            "configs/experiments/pilot_recovery_combined_deepseek_20.json"
        )
        self.assertEqual(combined["run_record_count"], 80)
        self.assertEqual(combined["worst_case_total_llm_call_count"], 100)
        self.assertNotIn("H2_pddl_recovery", combined["harness_modes"])

    def test_failure_memory_builder_freezes_only_successful_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_path = Path(tmpdir) / "tasks.jsonl"
            runs_path = Path(tmpdir) / "runs.jsonl"
            memory_path = Path(tmpdir) / "memory.jsonl"
            task = self.eval_tasks["eval_move_apple"]
            dump_jsonl(tasks_path, [task.to_dict()])
            runs_path.write_text(
                json.dumps(
                    {
                        "task_id": task.id,
                        "trace": {"final_state": ["on(apple, countertop)"]},
                        "attempts": [
                            {
                                "trace": {"violation": {"type": "missing_step"}},
                                "patch": {
                                    "source": "unit_repair",
                                    "before": ["pickup(apple)"],
                                    "after": list(task.gold_plan),
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = build_frozen_failure_memory(
                tasks_path=tasks_path,
                runs_path=runs_path,
                output_path=memory_path,
            )
            memory = FrozenFailureMemory.from_jsonl(memory_path)

            self.assertEqual(manifest["entry_count"], 1)
            self.assertEqual(len(memory.entries), 1)
            self.assertEqual(memory.entries[0].error_type, "missing_step")
            self.assertEqual(memory.sha256, manifest["sha256"])

    def test_runner_writes_complete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="unit_matrix",
                tasks_path="data/sample_tasks.jsonl",
                output_dir=tmpdir,
            )
            runner = ExperimentRunner(config)
            runs, records, summary = runner.run()
            self.assertEqual(len(runs), 36)
            self.assertEqual(len(records), 36)
            self.assertEqual(len(summary), 12)
            self.assertIsNotNone(runner.output_dir)
            manifest = json.loads(
                (runner.output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["data"]["tasks"]["task_count"], 5)
            self.assertIn("eval_move_apple", manifest["data"]["tasks"]["evaluation_task_ids"])
            self.assertIn("sha256", manifest["prompts"]["template"])
            analysis = json.loads(
                (runner.output_dir / "analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(analysis["record_count"], 36)
            self.assertEqual(analysis["method_count"], 12)
            self.assertIn("dataset", analysis["stratified"])
            self.assertTrue(analysis["paired_comparisons"])
            eai = manifest["code"]["submodules"]["external/embodied-agent-interface"]
            self.assertEqual(eai["pinned_commit"], eai["checked_out_commit"])

    def test_research_analysis_reports_ci_mcnemar_cost_and_search(self) -> None:
        low, high = wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

        mcnemar = exact_mcnemar(
            {"a": False, "b": False, "c": True},
            {"a": True, "b": True, "c": True},
        )
        self.assertEqual(mcnemar["right_only_success"], 2)
        self.assertEqual(mcnemar["left_only_success"], 0)
        self.assertEqual(mcnemar["exact_two_sided_p_value"], 0.5)

        metrics = [
            {
                "task_id": "a",
                "method_id": "method_a",
                "task_success": True,
                "safe_success": True,
                "execution_success": True,
                "risk": False,
                "attempts": 1,
                "patch_count": 0,
                "error_counts": {},
                "metadata": {"dataset": "virtualhome", "difficulty": "easy"},
            },
            {
                "task_id": "a",
                "method_id": "method_b",
                "task_success": False,
                "safe_success": False,
                "execution_success": False,
                "risk": False,
                "attempts": 2,
                "patch_count": 1,
                "error_counts": {"missing_step": 1},
                "metadata": {"dataset": "virtualhome", "difficulty": "easy"},
            },
        ]
        runs = [
            {
                "task_id": "a",
                "method_id": "method_a",
                "initial_plan": {
                    "metadata": {
                        "llm_call": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                            "latency_seconds": 1.5,
                            "estimated_cost_usd": 0.01,
                        },
                        "explored_states": 7,
                        "search_seconds": 0.2,
                    }
                },
                "patches": [],
            }
        ]
        report = build_research_analysis(metrics, runs)
        cost = report["cost_and_search"]["method_a"]
        self.assertEqual(cost["total_tokens"], 120)
        self.assertEqual(cost["symbolic_explored_states"], 7)
        self.assertEqual(cost["estimated_cost_per_success_usd"], 0.01)
        self.assertEqual(
            report["methods"]["method_b"]["failure_type_counts"]["missing_step"],
            1,
        )

    def test_one_api_client_records_usage_cost_and_prompt_fingerprint(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "id": "response-unit-1",
                        "choices": [
                            {
                                "message": {"content": '["noop()"]'},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                        },
                    }
                ).encode("utf-8")

        client = OneAPIChatClient(
            api_key="unit-secret",
            base_url="https://example.invalid/v1",
            model="unit-model",
            input_cost_per_million=2.0,
            output_cost_per_million=8.0,
        )
        with mock.patch(
            "embodied_gap.llm.clients.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            result = client.generate("unit prompt")

        self.assertEqual(result, '["noop()"]')
        call = client.last_call_metadata()
        self.assertEqual(call["total_tokens"], 150)
        self.assertEqual(call["response_id"], "response-unit-1")
        self.assertEqual(call["finish_reason"], "stop")
        self.assertAlmostEqual(call["estimated_cost_usd"], 0.0006)
        self.assertEqual(len(call["prompt_sha256"]), 64)
        self.assertNotIn("unit-secret", json.dumps(call))
        telemetry = client.telemetry()
        self.assertEqual(telemetry["call_count"], 1)
        self.assertEqual(telemetry["total_tokens"], 150)
        self.assertAlmostEqual(telemetry["estimated_cost_usd"], 0.0006)

    def test_one_api_client_lists_models(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {"id": "gpt-5.5", "object": "model"},
                            {"id": "Claude Sonnet 5", "object": "model"},
                            {"id": "gpt-5.5", "object": "model"},
                        ],
                    }
                ).encode("utf-8")

        client = OneAPIChatClient(
            api_key="unit-secret",
            base_url="https://example.invalid/v1",
            model="unit-model",
        )
        with mock.patch(
            "embodied_gap.llm.clients.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as request:
            models = client.list_models()

        self.assertEqual(models, ["Claude Sonnet 5", "gpt-5.5"])
        issued_request = request.call_args.args[0]
        self.assertEqual(issued_request.full_url, "https://example.invalid/v1/models")

    def test_runner_uses_external_retrieval_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = load_tasks("data/sample_tasks.jsonl")
            train_path = Path(tmpdir) / "rag_train.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"
            dump_jsonl(train_path, [task.to_dict() for task in tasks if task.split == "train"])
            dump_jsonl(
                eval_path,
                [task.to_dict() for task in tasks if task.id == "eval_move_apple"],
            )
            config = ExperimentConfig(
                name="unit_external_rag",
                tasks_path=str(eval_path),
                output_dir=str(Path(tmpdir) / "runs"),
                retrieval_examples_path=str(train_path),
                planners=("P1_retrieval_augmented",),
                harness_modes=("H0_open_loop",),
            )
            runs, records, _ = ExperimentRunner(config).run()
            self.assertTrue(records[0].task_success)
            self.assertEqual(runs[0].final_plan.metadata["retrieved"], "train_move_mug")

    def test_eai_pddl_parser_preserves_clean_problem_state(self) -> None:
        problem = parse_pddl_problem(
            """
            (define (problem Turn_on_light)
              (:domain virtualhome)
              (:objects character - character light room - object)
              (:init (off light) (inside character room) (not (on light)))
              (:goal (and (on light) (plugged_in light)))
            )
            """
        )
        self.assertEqual(problem.problem_name, "Turn_on_light")
        self.assertEqual(problem.objects["light"], "object")
        self.assertIn("off(light)", problem.init_facts)
        self.assertIn("not(on(light))", problem.init_facts)
        self.assertEqual(problem.goal_facts, ("on(light)", "plugged_in(light)"))
        self.assertEqual(format_plan_step("plug_in character light"), "plug_in(character, light)")

    def test_eai_adapter_imports_only_raw_source_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "src" / "virtualhome_eval" / "resources" / "virtualhome"
            problem_dir = dataset_root / "problem_pddl" / "Turn_on_light"
            problem_dir.mkdir(parents=True)
            (problem_dir / "11_1.pddl").write_text(
                """
                (define (problem Turn_on_light)
                  (:domain virtualhome)
                  (:objects character - character light room - object)
                  (:init (off light) (inside character room))
                  (:goal (and (on light)))
                )
                """,
                encoding="utf-8",
            )
            fixtures = {
                "id2task.json": {"11_1": "Turn on light"},
                "gold_pddl_plan.json": {
                    "11_1": ["walk_towards character light", "switch_on character light"]
                },
                "id2action.json": {"11_1": ["walk_towards", "switch_on"]},
                "id2predicate.json": {"11_1": ["off", "on", "inside"]},
                "success_task.json": ["11_1"],
                "failed_task.json": [],
            }
            for name, payload in fixtures.items():
                (dataset_root / name).write_text(json.dumps(payload), encoding="utf-8")
            (dataset_root / "virtualhome.pddl").write_text(
                "(define (domain virtualhome))",
                encoding="utf-8",
            )

            tasks = EmbodiedAgentInterfaceAdapter(root).load("virtualhome", train_ratio=0)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].source, "eai_raw_virtualhome")
            self.assertEqual(tasks[0].instruction, "Turn on light")
            self.assertEqual(tasks[0].gold_plan[-1], "switch_on(character, light)")
            self.assertEqual(tasks[0].metadata["executor_status"], "pddl_semantics_not_flattened")
            self.assertFalse(Path(tasks[0].metadata["source_root"]).is_absolute())
            self.assertEqual(
                tasks[0].metadata["domain_relative_path"],
                "src/virtualhome_eval/resources/virtualhome/virtualhome.pddl",
            )
            self.assertEqual(
                tasks[0].metadata["problem_relative_path"],
                "src/virtualhome_eval/resources/virtualhome/problem_pddl/Turn_on_light/11_1.pddl",
            )

    def test_resource_paths_replace_stale_absolute_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "embodied-agent-interface"
            dataset_root = root / "src" / "virtualhome_eval" / "resources" / "virtualhome"
            problem_path = dataset_root / "problem_pddl" / "Turn_on_light" / "11_1.pddl"
            problem_path.parent.mkdir(parents=True)
            problem_path.write_text("(define (problem Turn_on_light))", encoding="utf-8")
            domain_path = dataset_root / "virtualhome.pddl"
            domain_path.write_text("(define (domain virtualhome))", encoding="utf-8")
            task = Task.from_dict(
                {
                    "id": "portable_virtualhome",
                    "instruction": "Turn on light",
                    "initial_facts": [],
                    "goal_facts": [],
                    "allowed_actions": [],
                    "action_model": {},
                    "slots": {
                        "dataset": "virtualhome",
                        "task_family": "Turn_on_light",
                        "file_id": "11_1",
                    },
                    "metadata": {
                        "source_root": "/Users/another-user/stale/eai",
                        "domain_pddl_path": "/Users/another-user/stale/virtualhome.pddl",
                        "domain_relative_path": "src/virtualhome_eval/resources/virtualhome/virtualhome.pddl",
                        "problem_relative_path": "src/virtualhome_eval/resources/virtualhome/problem_pddl/Turn_on_light/11_1.pddl",
                    },
                }
            )
            with mock.patch.dict("os.environ", {"EAI_SOURCE_ROOT": str(root)}):
                self.assertEqual(resolve_domain_path(task), domain_path.resolve())
                self.assertEqual(resolve_problem_path(task), problem_path.resolve())

    def test_behavior_resource_paths_support_original_layout_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "embodied-agent-interface"
            resources = (
                root
                / "src"
                / "behavior_eval"
                / "evaluation"
                / "transition_modeling"
                / "resources"
            )
            resources.mkdir(parents=True)
            domain_path = resources / "behavior_new.pddl"
            domain_path.write_text("(define (domain igibson))", encoding="utf-8")
            problem_path = resources / "pddl_behavior" / "cleaning_test.pddl"
            problem_path.parent.mkdir()
            problem_path.write_text("(define (problem cleaning_test))", encoding="utf-8")
            task = Task.from_dict(
                {
                    "id": "portable_behavior",
                    "instruction": "Clean",
                    "initial_facts": [],
                    "goal_facts": [],
                    "allowed_actions": [],
                    "action_model": {},
                    "slots": {
                        "dataset": "behavior",
                        "task_family": "cleaning_test",
                        "file_id": "cleaning_test",
                    },
                    "metadata": {"source_root": "/Users/another-user/stale/eai"},
                }
            )
            with mock.patch.dict("os.environ", {"EAI_SOURCE_ROOT": str(root)}):
                self.assertEqual(resolve_domain_path(task), domain_path.resolve())
                self.assertEqual(resolve_problem_path(task), problem_path.resolve())

    def test_eai_adapter_refuses_historical_output_directories(self) -> None:
        with self.assertRaises(EAIAdapterError):
            EmbodiedAgentInterfaceAdapter("/tmp/output/diagnostics").load("virtualhome")

    def test_pddl_backed_executor_runs_grounded_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "mini.pddl"
            domain_path.write_text(
                """
                (define (domain mini)
                  (:requirements :typing)
                  (:types object)
                  (:predicates (next_to ?obj - object) (on ?obj - object) (open ?obj - object))
                  (:action switch_on
                    :parameters (?obj - object)
                    :precondition (next_to ?obj)
                    :effect (on ?obj))
                  (:action close
                    :parameters (?obj - object)
                    :precondition (open ?obj)
                    :effect (not (open ?obj)))
                )
                """,
                encoding="utf-8",
            )
            task = Task.from_dict(
                {
                    "id": "mini_pddl",
                    "instruction": "Turn on light",
                    "initial_facts": ["next_to(light)", "open(light)"],
                    "goal_facts": ["on(light)", "not(open(light))"],
                    "allowed_actions": ["switch_on(light)", "close(light)"],
                    "action_model": {},
                    "metadata": {
                        "executor_status": "pddl_semantics_not_flattened",
                        "domain_pddl_path": str(domain_path),
                        "objects": {"light": "object"},
                    },
                }
            )
            trace = SymbolicExecutor().execute(
                task,
                PlanCandidate("unit", ("switch_on(light)", "close(light)")),
            )
            self.assertEqual(trace.status, "success")
            self.assertEqual(trace.metadata["engine"], "pddl_backed")
            self.assertTrue(task.goal.is_satisfied(trace.final_state))

    def test_pddl_gold_validator_exports_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "mini.pddl"
            tasks_path = Path(tmpdir) / "tasks.jsonl"
            out_dir = Path(tmpdir) / "validation"
            domain_path.write_text(
                """
                (define (domain mini)
                  (:types object)
                  (:predicates (ready ?obj - object) (done ?obj - object))
                  (:action finish
                    :parameters (?obj - object)
                    :precondition (ready ?obj)
                    :effect (done ?obj))
                )
                """,
                encoding="utf-8",
            )
            tasks_path.write_text(
                json.dumps(
                    {
                        "id": "mini_gold",
                        "instruction": "finish item",
                        "initial_facts": ["ready(item)"],
                        "goal_facts": ["done(item)"],
                        "allowed_actions": ["finish(item)"],
                        "gold_plan": ["finish(item)"],
                        "action_model": {},
                        "slots": {"dataset": "unit", "task_family": "mini"},
                        "metadata": {
                            "executor_status": "pddl_semantics_not_flattened",
                            "domain_pddl_path": str(domain_path),
                            "objects": {"item": "object"},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = PDDLGoldPlanValidator().export(tasks_path, out_dir)
            self.assertEqual(summary["overall"]["success_rate"], 1.0)
            self.assertTrue((out_dir / "gold_plan_validation.jsonl").exists())

    def test_graph_grounded_planner_solves_pddl_inside_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "behavior_like.pddl"
            domain_path.write_text(
                """
                (define (domain behavior_like)
                  (:types object agent)
                  (:predicates
                    (inside ?obj - object ?container - object)
                    (open ?obj - object)
                    (holding ?obj - object)
                    (handsfull ?agent - agent)
                    (in_reach_of_agent ?obj - object))
                  (:action navigate_to
                    :parameters (?obj - object ?agent - agent)
                    :precondition (not (in_reach_of_agent ?obj))
                    :effect (in_reach_of_agent ?obj))
                  (:action open
                    :parameters (?obj - object ?agent - agent)
                    :precondition (and (in_reach_of_agent ?obj) (not (open ?obj)) (not (handsfull ?agent)))
                    :effect (open ?obj))
                  (:action grasp
                    :parameters (?obj - object ?agent - agent)
                    :precondition (and (in_reach_of_agent ?obj) (not (holding ?obj)) (not (handsfull ?agent)))
                    :effect (and (holding ?obj) (handsfull ?agent)))
                  (:action place_inside
                    :parameters (?obj - object ?container - object ?agent - agent)
                    :precondition (and (holding ?obj) (in_reach_of_agent ?container) (open ?container))
                    :effect (and (inside ?obj ?container) (not (holding ?obj)) (not (handsfull ?agent))))
                )
                """,
                encoding="utf-8",
            )
            task = Task.from_dict(
                {
                    "id": "mini_inside",
                    "instruction": "put gift in basket",
                    "initial_facts": [],
                    "goal_facts": ["inside(gift, basket)"],
                    "allowed_actions": ["navigate_to", "open", "grasp", "place_inside"],
                    "action_model": {},
                    "slots": {"dataset": "behavior", "task_family": "mini"},
                    "metadata": {
                        "executor_status": "pddl_semantics_not_flattened",
                        "domain_pddl_path": str(domain_path),
                        "objects": {"agent": "agent", "gift": "object", "basket": "object"},
                    },
                }
            )
            plan = GraphGroundedPlanner().plan(task)
            trace = SymbolicExecutor().execute(task, plan)
            self.assertTrue(plan.metadata["solved"])
            self.assertTrue(task.goal.is_satisfied(trace.final_state))

    def test_pddl_grounded_search_fallback_uses_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "search_only.pddl"
            domain_path.write_text(
                """
                (define (domain search_only)
                  (:types object)
                  (:predicates (ready ?obj - object) (done ?obj - object))
                  (:action prepare
                    :parameters (?obj - object)
                    :precondition ()
                    :effect (ready ?obj))
                  (:action finish
                    :parameters (?obj - object)
                    :precondition (ready ?obj)
                    :effect (done ?obj))
                )
                """,
                encoding="utf-8",
            )
            task = Task.from_dict(
                {
                    "id": "mini_search_only",
                    "instruction": "finish item",
                    "initial_facts": [],
                    "goal_facts": ["done(item)"],
                    "allowed_actions": ["prepare", "finish"],
                    "action_model": {},
                    "slots": {"dataset": "unit", "task_family": "search_only"},
                    "metadata": {
                        "executor_status": "pddl_semantics_not_flattened",
                        "domain_pddl_path": str(domain_path),
                        "objects": {"item": "object"},
                    },
                }
            )
            result = PDDLGroundedSearch(max_depth=3, max_expansions=20).search(task)
            self.assertTrue(result.solved)
            self.assertEqual(result.actions, ("prepare(item)", "finish(item)"))

    def test_failure_memory_patterns_label_balanced_failures(self) -> None:
        tasks = {
            task.slots.get("task_family"): task
            for task in load_tasks("data/processed/tasksets/balanced_eval_20.jsonl")
        }
        cleaning_patterns = {
            pattern.name for pattern in classify_failure_patterns(tasks["cleaning_microwave_oven"])
        }
        coffee_patterns = {
            pattern.name for pattern in classify_failure_patterns(tasks["Make_coffee"])
        }
        self.assertIn("behavior_negative_cleaning", cleaning_patterns)
        self.assertIn("virtualhome_appliance_surface_activation", coffee_patterns)

        planner = GraphGroundedPlanner()
        run = HarnessController().run(
            tasks["Make_coffee"],
            planner,
            HarnessMode.H2_FULL_RECOVERY,
        )
        record = evaluate_run(tasks["Make_coffee"], run)
        self.assertTrue(record.task_success)
        self.assertFalse(Path(str(run.trace.metadata["domain_path"])).is_absolute())
        self.assertIn(
            "virtualhome_appliance_surface_activation",
            run.final_plan.metadata["failure_memory_patterns"],
        )

    def test_h2_replans_failed_pddl_plan_with_grounded_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "behavior_like.pddl"
            domain_path.write_text(
                """
                (define (domain behavior_like)
                  (:types object agent)
                  (:predicates
                    (inside ?obj - object ?container - object)
                    (open ?obj - object)
                    (holding ?obj - object)
                    (handsfull ?agent - agent)
                    (in_reach_of_agent ?obj - object))
                  (:action navigate_to
                    :parameters (?obj - object ?agent - agent)
                    :precondition (not (in_reach_of_agent ?obj))
                    :effect (in_reach_of_agent ?obj))
                  (:action open
                    :parameters (?obj - object ?agent - agent)
                    :precondition (and (in_reach_of_agent ?obj) (not (open ?obj)) (not (handsfull ?agent)))
                    :effect (open ?obj))
                  (:action grasp
                    :parameters (?obj - object ?agent - agent)
                    :precondition (and (in_reach_of_agent ?obj) (not (holding ?obj)) (not (handsfull ?agent)))
                    :effect (and (holding ?obj) (handsfull ?agent)))
                  (:action place_inside
                    :parameters (?obj - object ?container - object ?agent - agent)
                    :precondition (and (holding ?obj) (in_reach_of_agent ?container) (open ?container))
                    :effect (and (inside ?obj ?container) (not (holding ?obj)) (not (handsfull ?agent))))
                )
                """,
                encoding="utf-8",
            )
            task = Task.from_dict(
                {
                    "id": "mini_h2_repair",
                    "instruction": "put gift in basket",
                    "initial_facts": [],
                    "goal_facts": ["inside(gift, basket)"],
                    "allowed_actions": ["navigate_to", "open", "grasp", "place_inside"],
                    "action_model": {},
                    "slots": {"dataset": "behavior", "task_family": "mini"},
                    "metadata": {
                        "executor_status": "pddl_semantics_not_flattened",
                        "domain_pddl_path": str(domain_path),
                        "objects": {"agent": "agent", "gift": "object", "basket": "object"},
                    },
                }
            )
            run = HarnessController().run(task, PromptOnlyPlanner(), HarnessMode.H2_FULL_RECOVERY)
            record = evaluate_run(task, run)
            self.assertTrue(record.task_success)
            self.assertTrue(run.patches)
            self.assertEqual(run.patches[-1].metadata["engine"], "pddl_grounded_search")

            broken_initial_plan = PlanCandidate(
                planner_name="P0_prompt_only",
                actions=(),
                raw_response="```json\n[]\n```",
                metadata={"parse_error": "invalid syntax"},
            )
            repaired_run = HarnessController().run(
                task,
                PromptOnlyPlanner(),
                HarnessMode.H2_FULL_RECOVERY,
                initial_plan=broken_initial_plan,
            )
            repaired_record = evaluate_run(task, repaired_run)
            self.assertTrue(repaired_record.task_success)
            self.assertNotIn("parse_error", repaired_run.final_plan.metadata)
            self.assertEqual(
                repaired_run.final_plan.metadata["repaired_from_parse_error"],
                "invalid syntax",
            )

    def test_taskset_builder_exports_balanced_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = TaskSetBuilder(list(self.eval_tasks.values()) + self.examples).export(
                tmpdir,
                per_family=1,
            )
            self.assertIn("balanced_eval", manifest["files"])
            self.assertIn("balanced_eval_20", manifest["files"])
            self.assertIn("balanced_eval_50", manifest["files"])
            self.assertIn("eai_smoke_eval", manifest["files"])
            self.assertTrue((Path(tmpdir) / "balanced_eval.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "balanced_eval_20.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "balanced_eval_50.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "eai_smoke_eval.jsonl").exists())
            self.assertGreaterEqual(manifest["files"]["full_eval"]["rows"], 1)
            difficulty = classify_difficulty(self.eval_tasks["eval_clean_plate"])
            self.assertIn(difficulty.label, {"easy", "medium", "hard"})

    def test_heldout_split_is_frozen_disjoint_and_virtualhome_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = freeze_heldout_split(
                executable_path="data/processed/tasksets/executable_eval.jsonl",
                development_path="data/processed/tasksets/balanced_eval.jsonl",
                output_dir=tmpdir,
                name="heldout_virtualhome_119",
                expected_count=119,
                expected_dataset="virtualhome",
            )
            repeated = freeze_heldout_split(
                executable_path="data/processed/tasksets/executable_eval.jsonl",
                development_path="data/processed/tasksets/balanced_eval.jsonl",
                output_dir=tmpdir,
                name="heldout_virtualhome_119",
                expected_count=119,
                expected_dataset="virtualhome",
            )
            heldout = load_tasks(Path(tmpdir) / "heldout_virtualhome_119.jsonl")
            development_ids = {
                task.id for task in load_tasks("data/processed/tasksets/balanced_eval.jsonl")
            }

            self.assertEqual(manifest, repeated)
            self.assertEqual(manifest["heldout_task_count"], 119)
            self.assertEqual(manifest["overlap_count"], 0)
            self.assertEqual(manifest["by_dataset"], {"virtualhome": 119})
            self.assertFalse({task.id for task in heldout} & development_ids)
            self.assertEqual(
                manifest["task_ids_sha256"],
                json.loads(
                    (Path(tmpdir) / "heldout_virtualhome_119_ids.json").read_text(
                        encoding="utf-8"
                    )
                )["task_ids_sha256"],
            )

            with self.assertRaises(ValueError):
                freeze_heldout_split(
                    executable_path="data/processed/tasksets/executable_eval.jsonl",
                    development_path="data/processed/tasksets/balanced_eval.jsonl",
                    output_dir=Path(tmpdir) / "wrong",
                    name="heldout_virtualhome_118",
                    expected_count=118,
                    expected_dataset="virtualhome",
                )

    def test_pddl_prompt_includes_action_signatures_without_gold_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "behavior_like.pddl"
            domain_path.write_text(
                """
                (define (domain behavior_like)
                  (:types object agent)
                  (:predicates
                    (inside ?obj - object ?container - object)
                    (holding ?obj - object)
                    (in_reach_of_agent ?obj - object))
                  (:action navigate_to
                    :parameters (?obj - object ?agent - agent)
                    :precondition (not (in_reach_of_agent ?obj))
                    :effect (in_reach_of_agent ?obj))
                  (:action place_inside
                    :parameters (?obj - object ?container - object ?agent - agent)
                    :precondition (holding ?obj)
                    :effect (inside ?obj ?container))
                )
                """,
                encoding="utf-8",
            )
            task = Task.from_dict(
                {
                    "id": "mini_prompt_schema",
                    "instruction": "put gift in basket",
                    "initial_facts": ["holding(gift)"],
                    "goal_facts": ["inside(gift, basket)"],
                    "allowed_actions": ["navigate_to", "place_inside"],
                    "gold_plan": ["place_inside(gift, basket, agent)"],
                    "action_model": {},
                    "metadata": {
                        "executor_status": "pddl_semantics_not_flattened",
                        "domain_pddl_path": str(domain_path),
                        "objects": {"agent": "agent", "gift": "object", "basket": "object"},
                    },
                }
            )
            prompt = render_planning_prompt(task, strategy="unit_test")
            self.assertIn("PDDL action signatures:", prompt)
            self.assertIn("navigate_to(obj:object, agent:agent)", prompt)
            self.assertIn("place_inside(obj:object, container:object, agent:agent)", prompt)
            self.assertIn("agent in [agent]", prompt)
            self.assertNotIn("place_inside(gift, basket, agent)", prompt)
            self.assertNotIn("gold_plan", prompt)

    def test_model_matrix_config_parses_model_overrides(self) -> None:
        config = ModelMatrixConfig.from_json("configs/experiments/sample_multimodel_one_api.json")
        self.assertEqual(config.name, "sample_multimodel_one_api")
        self.assertEqual(len(config.models), 3)
        self.assertEqual(config.models[0].model, "DeepSeek-V4-Flash")
        self.assertTrue(config.base_experiment.use_llm_for_planners)
        eai_config = ModelMatrixConfig.from_json(
            "configs/experiments/eai_balanced_20_multimodel_one_api.json"
        )
        self.assertEqual(eai_config.base_experiment.tasks_path, "data/processed/tasksets/balanced_eval_20.jsonl")
        self.assertEqual(eai_config.base_experiment.retrieval_examples_path, "data/processed/tasksets/rag_train.jsonl")
        self.assertEqual(len(eai_config.models), 2)

        model = ModelSpec.from_dict(
            {
                "id": "kimi",
                "model": "Kimi-K2.6",
                "temperature": 1,
                "max_tokens": 4096,
                "timeout_seconds": 240,
                "max_attempts": 2,
            }
        )
        self.assertEqual(model.temperature, 1.0)
        self.assertEqual(model.max_tokens, 4096)
        self.assertEqual(model.timeout_seconds, 240)
        self.assertEqual(model.max_attempts, 2)

        generalization = ModelMatrixConfig.from_json(
            "configs/experiments/eai_model_generalization_smoke.json"
        )
        self.assertEqual(len(generalization.models), 6)
        self.assertEqual(
            [item.model for item in generalization.models if item.enabled],
            ["DeepSeek-V4-Flash", "gpt-5.5", "DeepSeek-V4-Pro", "GLM-5-Turbo"],
        )
        kimi = next(item for item in generalization.models if item.id == "kimi_k2_6")
        self.assertEqual(kimi.temperature, 1.0)

        rag_config = ModelMatrixConfig.from_json(
            "configs/experiments/eai_planning_ablation_dev.json"
        ).base_experiment
        self.assertEqual(rag_config.retrieval_method, "lexical")
        self.assertEqual(
            rag_config.retrieval_field_profile,
            "instruction_state_goal_schema",
        )
        self.assertEqual(rag_config.retrieval_top_k, 1)
        self.assertEqual(rag_config.retrieval_min_score, 0.0)

    def test_experiment_runner_allocates_unique_non_overwriting_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="unit_unique_runs",
                tasks_path="data/sample_tasks.jsonl",
                output_dir=tmpdir,
                planners=("P0_prompt_only",),
                harness_modes=("H0_open_loop",),
            )
            first = ExperimentRunner(config)
            second = ExperimentRunner(config)
            first.run()
            second.run()
            self.assertNotEqual(first.output_dir, second.output_dir)
            self.assertTrue((first.output_dir / "summary.json").exists())
            self.assertTrue((second.output_dir / "summary.json").exists())
            index_rows = (Path(tmpdir) / "run_index.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(index_rows), 2)

            occupied = Path(tmpdir) / "occupied"
            occupied.mkdir()
            marker = occupied / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ExperimentRunner(config, output_dir=occupied).run()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_model_matrix_rerun_preserves_previous_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = ExperimentConfig(
                name="unit_model_matrix_history",
                tasks_path="data/sample_tasks.jsonl",
                output_dir=tmpdir,
                use_llm_for_planners=False,
            )
            first_config = ModelMatrixConfig(
                name="unit_model_matrix_history",
                base_experiment=base,
                models=(
                    ModelSpec(id="deepseek", model="deterministic-deepseek"),
                    ModelSpec(id="gpt", model="deterministic-gpt"),
                ),
            )
            first = MultiModelExperimentRunner(first_config).run()
            deepseek_summary = Path(first["models"]["deepseek"]["output_dir"]) / "summary.json"
            self.assertTrue(deepseek_summary.exists())
            preserved = deepseek_summary.read_text(encoding="utf-8")

            second_config = ModelMatrixConfig(
                name="unit_model_matrix_history",
                base_experiment=base,
                models=(ModelSpec(id="gpt", model="deterministic-gpt"),),
            )
            second = MultiModelExperimentRunner(second_config).run()
            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertTrue(deepseek_summary.exists())
            self.assertEqual(deepseek_summary.read_text(encoding="utf-8"), preserved)
            self.assertNotIn("deepseek", second["models"])

    def test_model_matrix_failed_rerun_preserves_prior_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            successful = ModelMatrixConfig(
                name="unit_model_matrix_failure_history",
                base_experiment=ExperimentConfig(
                    name="unit_model_matrix_failure_history",
                    tasks_path="data/sample_tasks.jsonl",
                    output_dir=tmpdir,
                    use_llm_for_planners=False,
                ),
                models=(ModelSpec(id="model", model="deterministic"),),
            )
            first = MultiModelExperimentRunner(successful).run()
            prior_summary = Path(first["models"]["model"]["output_dir"]) / "summary.json"
            prior_content = prior_summary.read_text(encoding="utf-8")

            failing = ModelMatrixConfig(
                name="unit_model_matrix_failure_history",
                base_experiment=ExperimentConfig(
                    name="unit_model_matrix_failure_history",
                    tasks_path="data/processed/tasksets/does_not_exist.jsonl",
                    output_dir=tmpdir,
                    use_llm_for_planners=False,
                ),
                models=(ModelSpec(id="model", model="deterministic"),),
            )
            second = MultiModelExperimentRunner(failing).run()
            self.assertEqual(second["failed"], 1)
            self.assertEqual(second["succeeded"], 0)
            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertTrue(prior_summary.exists())
            self.assertEqual(prior_summary.read_text(encoding="utf-8"), prior_content)
            failed_model_dir = Path(second["models"]["model"]["output_dir"])
            self.assertTrue((failed_model_dir / "error.json").exists())
            failed_manifest = json.loads(
                (failed_model_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_manifest["status"], "failed")
            matrix_manifest = json.loads(
                (Path(second["output_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(matrix_manifest["status"], "partial")

    def test_knowledge_builder_exports_rag_docs_and_kg_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = KnowledgeCorpusBuilder(self.examples).export(tmpdir)
            self.assertGreater(manifest["retrieval_doc_count"], 0)
            self.assertGreater(manifest["kg_edge_count"], 0)
            self.assertTrue((Path(tmpdir) / "retrieval_corpus.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "kg_edges.jsonl").exists())

    def test_parser_accepts_action_dictionary_formats(self) -> None:
        self.assertEqual(
            parse_action_list('[{"action":"grasp","args":["cup","agent"]}]'),
            ("grasp(cup, agent)",),
        )
        self.assertEqual(
            parse_action_list('[{"action":"grasp","object":"cup, agent"}]'),
            ("grasp(cup, agent)",),
        )
        self.assertEqual(
            parse_action_list('```json\n["grasp(cup, agent)"]\n```'),
            ("grasp(cup, agent)",),
        )
        self.assertEqual(
            parse_action_list('[["grasp","cup","agent"]]'),
            ("grasp(cup, agent)",),
        )
        self.assertEqual(
            parse_action_list('[["grasp",["cup","agent"]]]'),
            ("grasp(cup, agent)",),
        )


if __name__ == "__main__":
    unittest.main()
