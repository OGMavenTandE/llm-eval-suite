import argparse
import sys

from llm_eval.core.config_service import ConfigService
from llm_eval.core.run_service import RunService


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

    config_service = ConfigService()
    run_service = RunService(config_service=config_service)

    try:
        config = config_service.load_yaml(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: Failed to parse YAML config '{args.config}': {exc}", file=sys.stderr)
        sys.exit(1)

    structure_errors = config_service.validate_config_structure(config)
    if structure_errors:
        config_service.exit_on_errors(structure_errors)

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    from llm_eval.offline import activate_run_offline

    activate_run_offline(config=config)
    config["_config_path"] = args.config

    if args.dry_run:
        result = run_service.run_dry_run(config, verbose=args.verbose)
        if result.status == "failed_validation":
            config_service.exit_on_errors(result.validation.errors if result.validation else [])
        if result.status in ("failed_runtime", "failed"):
            print(f"Error: {result.error_message or result.message}", file=sys.stderr)
            sys.exit(1)
        return

    result = run_service.start_run(config, verbose=args.verbose, compare=args.compare)
    if result.status == "failed_validation":
        config_service.exit_on_errors(result.validation.errors if result.validation else [])
    if result.status in ("failed_runtime", "failed"):
        print(f"Error: {result.error_message or result.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
