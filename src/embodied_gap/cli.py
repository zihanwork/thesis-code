from __future__ import annotations

import argparse
import json

from embodied_gap.analysis.make_tables import summary_to_markdown
from embodied_gap.datasets.eai_adapter import EmbodiedAgentInterfaceAdapter
from embodied_gap.datasets.taskset_builder import TaskSetBuilder
from embodied_gap.experiments.config import ExperimentConfig
from embodied_gap.experiments.model_matrix import ModelMatrixConfig, MultiModelExperimentRunner
from embodied_gap.experiments.runner import ExperimentRunner
from embodied_gap.evaluation.pddl_gold_validator import PDDLGoldPlanValidator
from embodied_gap.knowledge.corpus_builder import KnowledgeCorpusBuilder
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
    client = OneAPIChatClient.from_env(args.env)
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
    check_parser.set_defaults(func=check_one_api)

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
