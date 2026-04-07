import argparse
import sys

from llm_eval.runner import EvalRunner


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print(
            "Error: pyyaml is not installed. Install it with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: Failed to parse YAML config '{path}': {exc}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_eval",
        description="LLM Evaluation Suite — run evaluations from a YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML evaluation config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Override the output directory specified in the config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and dataset without running inference.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help=(
            "Enable comparison mode. Automatically active when multiple models are "
            "configured; use this flag to force it even with a single model (useful "
            "for comparing against future runs)."
        ),
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _load_yaml(args.config)

    if not isinstance(config, dict):
        print(
            f"Error: YAML config must be a mapping (got {type(config).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)

    required_keys = ["models", "evaluators", "dataset"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        print(
            f"Error: Config is missing required key(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # CLI overrides
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    runner = EvalRunner(config=config, dry_run=args.dry_run, verbose=args.verbose, compare=args.compare)
    runner.run()


if __name__ == "__main__":
    main()
