import logging
import traceback
from datetime import datetime

from llm_eval.datasets import loader as dataset_loader
from llm_eval.reporting.reporter import EvalReporter
from llm_eval.reporting.comparison import ComparisonReporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry — maps provider name -> adapter class
# ---------------------------------------------------------------------------

def _get_model_class(provider: str):
    if provider == "ollama":
        from llm_eval.models.ollama_model import OllamaModel
        return OllamaModel
    elif provider == "openai":
        from llm_eval.models.openai_model import OpenAIModel
        return OpenAIModel
    else:
        raise ValueError(f"Unknown model provider: '{provider}'. Supported: ollama, openai")


# ---------------------------------------------------------------------------
# Evaluator registry — maps evaluator name -> evaluator class
# ---------------------------------------------------------------------------

def _get_evaluator_class(name: str):
    if name == "correctness":
        from llm_eval.evaluators.correctness import CorrectnessEvaluator
        return CorrectnessEvaluator
    elif name == "latency":
        from llm_eval.evaluators.latency import LatencyEvaluator
        return LatencyEvaluator
    elif name == "robustness":
        from llm_eval.evaluators.robustness import RobustnessEvaluator
        return RobustnessEvaluator
    elif name == "consistency":
        from llm_eval.evaluators.consistency import ConsistencyEvaluator
        return ConsistencyEvaluator
    elif name == "cost":
        from llm_eval.evaluators.cost import CostEvaluator
        return CostEvaluator
    else:
        raise ValueError(
            f"Unknown evaluator: '{name}'. Supported: correctness, latency, robustness, consistency, cost"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """
    Orchestrates a full evaluation run: loads dataset and models, runs each
    evaluator over every sample, and writes results via EvalReporter.

    When multiple models are configured (or --compare is passed), a
    ComparisonReporter is also generated after all per-model runs complete.
    """

    def __init__(self, config: dict, dry_run: bool = False, verbose: bool = False, compare: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose
        self.compare = compare

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    def run(self):
        config = self.config
        run_name = config.get("run_name", "eval_run")
        output_dir = config.get("output_dir", "results/")
        dataset_path = config["dataset"]
        models_cfg = config.get("models", [])
        evaluators_cfg = config.get("evaluators", [])

        # --- Load dataset ---
        logger.info(f"Loading dataset: {dataset_path}")
        samples = dataset_loader.load_dataset(dataset_path)
        n = len(samples)
        logger.info(f"Loaded {n} samples.")

        # --- Validate model and evaluator configs ---
        for model_cfg in models_cfg:
            _get_model_class(model_cfg["provider"])  # raises on unknown provider

        for eval_cfg in evaluators_cfg:
            _get_evaluator_class(eval_cfg["name"])  # raises on unknown evaluator

        if self.dry_run:
            print(f"[dry-run] Config valid. {n} samples, {len(models_cfg)} model(s), "
                  f"{len(evaluators_cfg)} evaluator(s). No inference will run.")
            return

        # --- Instantiate evaluators once (shared across models) ---
        evaluators = []
        for eval_cfg in evaluators_cfg:
            cls = _get_evaluator_class(eval_cfg["name"])
            evaluators.append(cls(eval_cfg))

        # Identify which evaluators need special kwargs by class name.
        # Using isinstance checks is more reliable than signature introspection
        # because some evaluators access kwargs via **kwargs rather than named params.
        from llm_eval.evaluators.latency import LatencyEvaluator
        from llm_eval.evaluators.correctness import CorrectnessEvaluator
        from llm_eval.evaluators.consistency import ConsistencyEvaluator
        from llm_eval.evaluators.cost import CostEvaluator
        from llm_eval.evaluators.robustness import RobustnessEvaluator

        # Find the correctness evaluator (needed by robustness evaluator)
        correctness_evaluator = None
        for ev in evaluators:
            if isinstance(ev, CorrectnessEvaluator):
                correctness_evaluator = ev
                break

        # Decide whether to run comparison mode
        run_comparison = self.compare or len(models_cfg) > 1

        # Collect per-model reporters for comparison
        model_reporters = {}  # model_name -> EvalReporter

        # --- Run per-model ---
        for model_cfg in models_cfg:
            model_name = model_cfg["name"]
            provider = model_cfg["provider"]
            model_run_name = f"{run_name}_{model_name.replace(':', '_').replace('/', '_')}"

            logger.info(f"\n=== Model: {model_name} (provider: {provider}) ===")

            model_cls = _get_model_class(provider)
            model = model_cls(model_cfg["name"], model_cfg.get("params", {}))

            reporter = EvalReporter(output_dir=output_dir, run_name=model_run_name)

            # Determine judge model for llm_judge correctness (use same model by default)
            judge_model = None
            for eval_cfg in evaluators_cfg:
                if eval_cfg.get("name") == "correctness" and eval_cfg.get("mode") == "llm_judge":
                    judge_provider = eval_cfg.get("judge_provider", provider)
                    if judge_provider == provider:
                        judge_model = model
                    else:
                        judge_cls = _get_model_class(judge_provider)
                        judge_cfg = eval_cfg.get("judge_model_config", model_cfg)
                        judge_model = judge_cls(judge_cfg["name"], judge_cfg.get("params", {}))
                    break

            # --- Iterate samples ---
            for idx, sample in enumerate(samples):
                print(f"Evaluating sample {idx + 1}/{n}...")
                prompt = sample["prompt"]
                expected = sample["expected_answer"]

                try:
                    response = model.generate(prompt)
                    response_text = response.text
                    latency_ms = response.latency_ms
                    tokens_used = response.tokens_used
                except Exception as exc:
                    logger.error(f"  [sample {idx}] Model error: {exc}")
                    if self.verbose:
                        traceback.print_exc()
                    continue

                eval_results = []
                for evaluator in evaluators:
                    try:
                        kwargs = {}

                        if isinstance(evaluator, LatencyEvaluator):
                            kwargs["latency_ms"] = latency_ms
                        elif isinstance(evaluator, CorrectnessEvaluator):
                            if judge_model is not None:
                                kwargs["judge_model"] = judge_model
                        elif isinstance(evaluator, ConsistencyEvaluator):
                            kwargs["model"] = model
                        elif isinstance(evaluator, CostEvaluator):
                            kwargs["tokens_used"] = tokens_used
                            kwargs["latency_ms"] = latency_ms
                        elif isinstance(evaluator, RobustnessEvaluator):
                            kwargs["model"] = model
                            if correctness_evaluator is not None:
                                kwargs["correctness_evaluator"] = correctness_evaluator
                            else:
                                logger.warning(
                                    "  Robustness evaluator requires a correctness evaluator. "
                                    "Add a 'correctness' evaluator before 'robustness' in your config."
                                )
                                continue

                        result = evaluator.evaluate(prompt, expected, response_text, **kwargs)
                        eval_results.append(result)

                        if self.verbose:
                            logger.debug(
                                f"  [{result.metric_name}] score={result.score:.4f} "
                                f"passed={result.passed} details={result.details}"
                            )
                    except Exception as exc:
                        logger.error(
                            f"  [sample {idx}] Evaluator '{evaluator.__class__.__name__}' error: {exc}"
                        )
                        if self.verbose:
                            traceback.print_exc()

                reporter.record_result(
                    sample_idx=idx,
                    prompt=prompt,
                    expected=expected,
                    model_response_text=response_text,
                    latency_ms=latency_ms,
                    eval_results=eval_results,
                )

            # --- Save outputs ---
            detailed_path = reporter.save_detailed_results()
            summary_path = reporter.save_summary()
            logger.info(f"Detailed results: {detailed_path}")
            logger.info(f"Summary CSV:      {summary_path}")
            reporter.print_summary()

            model_reporters[model_name] = reporter

        # --- Comparison mode ---
        if run_comparison and len(model_reporters) > 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_dir = f"{output_dir}/{run_name}_comparison_{timestamp}"
            model_names = list(model_reporters.keys())

            comp_reporter = ComparisonReporter(
                output_dir=comparison_dir,
                run_name=run_name,
                model_names=model_names,
            )

            for model_name, reporter in model_reporters.items():
                comp_reporter.add_model_results(model_name, reporter._results)

            comp_reporter.save_comparison()
            comp_reporter.print_comparison()
