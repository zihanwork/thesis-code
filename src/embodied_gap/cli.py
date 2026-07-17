from __future__ import annotations

import argparse
import json

from embodied_gap.analysis.make_tables import summary_to_markdown
from embodied_gap.analysis.model_generalization import (
    export_model_generalization_summary,
)
from embodied_gap.analysis.research_report import export_research_analysis
from embodied_gap.datasets.eai_adapter import EmbodiedAgentInterfaceAdapter
from embodied_gap.datasets.taskset_builder import TaskSetBuilder
from embodied_gap.datasets.split_freezer import freeze_heldout_split
from embodied_gap.datasets.safety_set import (
    export_frozen_safety_set,
    verify_frozen_safety_set,
)
from embodied_gap.experiments.config import ExperimentConfig
from embodied_gap.experiments.model_matrix import ModelMatrixConfig, MultiModelExperimentRunner
from embodied_gap.experiments.pilot_budget import inspect_model_matrix
from embodied_gap.experiments.runner import ExperimentRunner
from embodied_gap.evaluation.pddl_gold_validator import PDDLGoldPlanValidator
from embodied_gap.evaluation.official_eai import (
    export_official_preflight,
    export_virtualhome_action_sequencing,
)
from embodied_gap.evaluation.safety_benchmark import run_safety_benchmark
from embodied_gap.harness.recovery_policy import HarnessMode
from embodied_gap.knowledge.corpus_builder import KnowledgeCorpusBuilder
from embodied_gap.knowledge.failure_memory_store import build_frozen_failure_memory
from embodied_gap.llm.clients import OneAPIChatClient


def run(args: argparse.Namespace) -> int:
    if args.config:
        config = ExperimentConfig.from_json(args.config)
    else:
        config = ExperimentConfig(
            name=args.name,
            tasks_path=args.tasks,
            output_dir=args.out,
            max_retries=args.max_retries,
        )
    runner = ExperimentRunner(config)
    _, _, summary = runner.run()
    print(summary_to_markdown(summary))
    print(f"\nWrote experiment artifacts to {runner.output_dir}")
    return 0


def check_one_api(args: argparse.Namespace) -> int:
    client = OneAPIChatClient.from_env(
        args.env,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("Model:", client.model)
    try:
        response = client.generate('Return exactly this JSON list and nothing else: ["noop()"]')
    except RuntimeError as exc:
        print("One API smoke test: failed")
        print(str(exc))
        return 1
    else:
        print("One API smoke test: success")
        print("Response preview:", response[:120].replace("\n", " "))
        return 0


def list_one_api_models(args: argparse.Namespace) -> int:
    client = OneAPIChatClient.from_env(args.env)
    try:
        models = client.list_models()
    except RuntimeError as exc:
        print("One API model discovery: failed")
        print(str(exc))
        return 1
    print(json.dumps({"count": len(models), "models": models}, ensure_ascii=False, indent=2))
    return 0


def prepare_eai(args: argparse.Namespace) -> int:
    datasets = tuple(args.datasets.split(","))
    adapter = EmbodiedAgentInterfaceAdapter(args.source_root)
    manifest = adapter.export(
        out_dir=args.out_dir,
        datasets=datasets,
        train_ratio=args.train_ratio,
        write_raw=not args.no_raw_pddl,
    )
    print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nWrote clean EAI data to {args.out_dir}")
    return 0


def build_tasksets(args: argparse.Namespace) -> int:
    manifest = TaskSetBuilder.from_jsonl(args.tasks).export(
        out_dir=args.out_dir,
        per_family=args.per_family,
    )
    print(json.dumps(manifest["files"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nWrote task sets to {args.out_dir}")
    return 0


def run_model_matrix(args: argparse.Namespace) -> int:
    matrix_config = ModelMatrixConfig.from_json(args.config)
    summary = MultiModelExperimentRunner(matrix_config).run()
    compact = {
        model_id: {
            "status": payload["status"],
            "output_dir": payload["output_dir"],
            "method_count": len(payload.get("summary", {})),
            "error_type": payload.get("error_type"),
        }
        for model_id, payload in summary["models"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nWrote model matrix artifacts to {summary['output_dir']}")
    return 0


def inspect_matrix(args: argparse.Namespace) -> int:
    report = inspect_model_matrix(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def analyze_run(args: argparse.Namespace) -> int:
    report = export_research_analysis(
        metrics_path=args.metrics,
        runs_path=args.runs,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "record_count": report["record_count"],
                "method_count": report["method_count"],
                "comparison_count": len(report["paired_comparisons"]),
                "output_path": args.out,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def summarize_model_generalization(args: argparse.Namespace) -> int:
    report = export_model_generalization_summary(
        run_dir=args.run_dir,
        output_path=args.out,
        analysis_filename=args.analysis_filename,
    )
    print(
        json.dumps(
            {
                "model_count": report["matrix"]["model_count"],
                "run_id": report["source"]["run_id"],
                "dirty_worktree": report["source"]["dirty_worktree"],
                "output_path": args.out,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_knowledge(args: argparse.Namespace) -> int:
    manifest = KnowledgeCorpusBuilder.from_jsonl(args.tasks).export(args.out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nWrote RAG/KG artifacts to {args.out_dir}")
    return 0


def validate_pddl_gold(args: argparse.Namespace) -> int:
    summary = PDDLGoldPlanValidator().export(
        tasks_path=args.tasks,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nWrote gold-plan validation artifacts to {args.out_dir}")
    return 0


def build_failure_memory(args: argparse.Namespace) -> int:
    manifest = build_frozen_failure_memory(
        tasks_path=args.tasks,
        runs_path=args.runs,
        output_path=args.out,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def freeze_heldout(args: argparse.Namespace) -> int:
    manifest = freeze_heldout_split(
        executable_path=args.executable,
        development_path=args.development,
        output_dir=args.out_dir,
        name=args.name,
        expected_count=args.expected_count,
        expected_dataset=args.expected_dataset,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def check_official_eai(args: argparse.Namespace) -> int:
    report = export_official_preflight(
        args.responses,
        args.out,
        external_root=args.external_root,
    )
    print(
        json.dumps(
            {
                "present_slot_count": report["present_slot_count"],
                "required_slot_count": report["required_slot_count"],
                "action_sequencing_shapes_valid": report["action_sequencing_shapes_valid"],
                "structurally_ready": report["structurally_ready"],
                "official_runtime_ready": report["official_runtime_ready"],
                "submission_ready": report["submission_ready"],
                "output_path": args.out,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["action_sequencing_shapes_valid"] else 1


def export_official_virtualhome(args: argparse.Namespace) -> int:
    manifest = export_virtualhome_action_sequencing(
        runs_path=args.runs,
        tasks_path=args.tasks,
        prompts_path=args.prompts,
        output_path=args.out,
        planner_name=args.planner,
        harness_mode=args.harness,
        allow_partial=args.allow_partial,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest["complete"] else 1


def build_safety_set(args: argparse.Namespace) -> int:
    manifest = export_frozen_safety_set(args.out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def verify_safety_set(args: argparse.Namespace) -> int:
    report = verify_frozen_safety_set(args.tasks, args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def evaluate_safety_set(args: argparse.Namespace) -> int:
    modes = tuple(HarnessMode(item.strip()) for item in args.harness_modes.split(","))
    output_dir, summary = run_safety_benchmark(
        tasks_path=args.tasks,
        output_root=args.out,
        modes=modes,
        max_retries=args.max_retries,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "record_count": summary["record_count"],
                "method_count": summary["method_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodied-gap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the P/H experimental matrix.")
    run_parser.add_argument("--config")
    run_parser.add_argument("--name", default="sample_matrix")
    run_parser.add_argument("--tasks", default="data/sample_tasks.jsonl")
    run_parser.add_argument("--out", default="runs/sample_matrix")
    run_parser.add_argument("--max-retries", type=int, default=3)
    run_parser.set_defaults(func=run)

    check_parser = subparsers.add_parser("check-one-api", help="Check One API connectivity.")
    check_parser.add_argument("--env", default=".env")
    check_parser.add_argument(
        "--model",
        help="Override ONE_API_MODEL for a one-call compatibility check.",
    )
    check_parser.add_argument("--temperature", type=float)
    check_parser.add_argument("--max-tokens", type=int)
    check_parser.set_defaults(func=check_one_api)

    list_models_parser = subparsers.add_parser(
        "list-one-api-models",
        help="List model IDs exposed by the configured One API account.",
    )
    list_models_parser.add_argument("--env", default=".env")
    list_models_parser.set_defaults(func=list_one_api_models)

    eai_parser = subparsers.add_parser(
        "prepare-eai",
        help="Convert clean EAI raw benchmark resources into project JSONL data.",
    )
    eai_parser.add_argument(
        "--source-root",
        default="external/embodied-agent-interface",
        help=(
            "EAI checkout root. Defaults to the repository-local external clone; "
            "set EAI_SOURCE_ROOT or pass this option for an external checkout."
        ),
    )
    eai_parser.add_argument("--out-dir", default="data/processed/eai_clean")
    eai_parser.add_argument(
        "--datasets",
        default="virtualhome,behavior",
        help="Comma-separated EAI datasets to import: virtualhome,behavior.",
    )
    eai_parser.add_argument("--train-ratio", type=float, default=0.2)
    eai_parser.add_argument("--no-raw-pddl", action="store_true")
    eai_parser.set_defaults(func=prepare_eai)

    tasksets_parser = subparsers.add_parser(
        "build-tasksets",
        help="Build thesis-ready task subsets from clean canonical tasks.",
    )
    tasksets_parser.add_argument("--tasks", default="data/processed/eai_clean/all_tasks.jsonl")
    tasksets_parser.add_argument("--out-dir", default="data/processed/tasksets")
    tasksets_parser.add_argument("--per-family", type=int, default=8)
    tasksets_parser.set_defaults(func=build_tasksets)

    model_matrix_parser = subparsers.add_parser(
        "run-model-matrix",
        help="Run the same experiment matrix across multiple model names.",
    )
    model_matrix_parser.add_argument("--config", required=True)
    model_matrix_parser.set_defaults(func=run_model_matrix)

    inspect_matrix_parser = subparsers.add_parser(
        "inspect-model-matrix",
        help="Validate a model matrix and estimate its worst-case API-call budget.",
    )
    inspect_matrix_parser.add_argument("--config", required=True)
    inspect_matrix_parser.set_defaults(func=inspect_matrix)

    analysis_parser = subparsers.add_parser(
        "analyze-run",
        help="Generate confidence intervals, paired tests, strata, and cost summaries.",
    )
    analysis_parser.add_argument("--metrics", required=True)
    analysis_parser.add_argument("--runs")
    analysis_parser.add_argument("--out", required=True)
    analysis_parser.set_defaults(func=analyze_run)

    generalization_parser = subparsers.add_parser(
        "summarize-model-generalization",
        help="Summarize a completed multi-model development run.",
    )
    generalization_parser.add_argument("--run-dir", required=True)
    generalization_parser.add_argument("--out", required=True)
    generalization_parser.add_argument(
        "--analysis-filename",
        default="analysis_v2.json",
    )
    generalization_parser.set_defaults(func=summarize_model_generalization)

    knowledge_parser = subparsers.add_parser(
        "build-knowledge",
        help="Build retrieval corpus and KG edges from canonical tasks.",
    )
    knowledge_parser.add_argument("--tasks", default="data/processed/tasksets/rag_train.jsonl")
    knowledge_parser.add_argument("--out-dir", default="data/knowledge/eai_train")
    knowledge_parser.set_defaults(func=build_knowledge)

    validate_parser = subparsers.add_parser(
        "validate-pddl-gold",
        help="Validate canonical tasks by executing their gold PDDL plans.",
    )
    validate_parser.add_argument("--tasks", default="data/processed/tasksets/executable_eval.jsonl")
    validate_parser.add_argument("--out-dir", default="runs/pddl_gold_validation")
    validate_parser.add_argument("--limit", type=int)
    validate_parser.set_defaults(func=validate_pddl_gold)

    memory_parser = subparsers.add_parser(
        "build-failure-memory",
        help="Freeze successful development failure-to-repair pairs as read-only memory.",
    )
    memory_parser.add_argument("--tasks", required=True)
    memory_parser.add_argument("--runs", required=True, nargs="+")
    memory_parser.add_argument("--out", required=True)
    memory_parser.set_defaults(func=build_failure_memory)

    heldout_parser = subparsers.add_parser(
        "freeze-heldout",
        help="Freeze an executable-minus-development held-out split with hashes.",
    )
    heldout_parser.add_argument("--executable", required=True)
    heldout_parser.add_argument("--development", required=True)
    heldout_parser.add_argument("--out-dir", required=True)
    heldout_parser.add_argument("--name", required=True)
    heldout_parser.add_argument("--expected-count", required=True, type=int)
    heldout_parser.add_argument("--expected-dataset")
    heldout_parser.set_defaults(func=freeze_heldout)

    official_parser = subparsers.add_parser(
        "check-official-eai",
        help="Check official EAI response slots and action-sequencing formats.",
    )
    official_parser.add_argument("--responses", required=True)
    official_parser.add_argument(
        "--external-root",
        default="external/embodied-agent-interface",
    )
    official_parser.add_argument("--out", required=True)
    official_parser.set_defaults(func=check_official_eai)

    export_vh_parser = subparsers.add_parser(
        "export-official-virtualhome",
        help="Export one exact project method to official VirtualHome action-sequencing format.",
    )
    export_vh_parser.add_argument("--runs", required=True)
    export_vh_parser.add_argument("--tasks", required=True)
    export_vh_parser.add_argument(
        "--prompts",
        default=(
            "external/embodied-agent-interface/src/virtualhome_eval/evaluation/"
            "action_sequencing/prompts/helm_prompts.json"
        ),
    )
    export_vh_parser.add_argument("--planner", required=True)
    export_vh_parser.add_argument("--harness", required=True)
    export_vh_parser.add_argument("--out", required=True)
    export_vh_parser.add_argument("--allow-partial", action="store_true")
    export_vh_parser.add_argument("--overwrite", action="store_true")
    export_vh_parser.set_defaults(func=export_official_virtualhome)

    safety_set_parser = subparsers.add_parser(
        "build-safety-set",
        help="Create the non-overwriting frozen controlled safety set v1.",
    )
    safety_set_parser.add_argument(
        "--out-dir",
        default="data/processed/tasksets",
    )
    safety_set_parser.set_defaults(func=build_safety_set)

    safety_verify_parser = subparsers.add_parser(
        "verify-safety-set",
        help="Verify the frozen safety set against its hash, manifest, and v1 definition.",
    )
    safety_verify_parser.add_argument(
        "--tasks",
        default="data/processed/tasksets/safety_frozen_v1.jsonl",
    )
    safety_verify_parser.add_argument(
        "--manifest",
        default="data/processed/tasksets/safety_frozen_v1_manifest.json",
    )
    safety_verify_parser.set_defaults(func=verify_safety_set)

    safety_eval_parser = subparsers.add_parser(
        "evaluate-safety-set",
        help="Evaluate verifier and local recovery on frozen injected safety plans.",
    )
    safety_eval_parser.add_argument(
        "--tasks",
        default="data/processed/tasksets/safety_frozen_v1.jsonl",
    )
    safety_eval_parser.add_argument("--out", default="runs/safety_benchmark")
    safety_eval_parser.add_argument(
        "--harness-modes",
        default="H0_open_loop,H1_verifier_gated,H2_local_recovery",
    )
    safety_eval_parser.add_argument("--max-retries", type=int, default=3)
    safety_eval_parser.set_defaults(func=evaluate_safety_set)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
