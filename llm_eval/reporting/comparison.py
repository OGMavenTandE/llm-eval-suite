import csv
import json
from pathlib import Path


class ComparisonReporter:
    """
    Collects per-model results from multiple EvalReporter runs and generates
    side-by-side comparison outputs:

    - comparison_summary.csv  — model_name, metric_name, mean_score, pass_rate
    - comparison_detailed.json — per-sample side-by-side view across all models
    - stdout table             — formatted side-by-side metric scores
    """

    def __init__(self, output_dir: str, run_name: str, model_names: list):
        self.run_dir = Path(output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.model_names = model_names
        # {model_name: [result_record, ...]}
        self._model_results = {}

    def add_model_results(self, model_name: str, results_list: list):
        """Store the per-sample results list for a given model."""
        self._model_results[model_name] = results_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(self):
        """
        Returns a nested dict:
          {model_name: {metric_name: {"mean_score": float, "pass_rate": float}}}
        """
        agg = {}
        for model_name, results in self._model_results.items():
            metric_data = {}
            for record in results:
                for ev in record.get("evaluations", []):
                    name = ev["metric_name"]
                    if name not in metric_data:
                        metric_data[name] = {"scores": [], "passed": []}
                    metric_data[name]["scores"].append(ev["score"])
                    metric_data[name]["passed"].append(ev["passed"])

            agg[model_name] = {}
            for metric_name, data in metric_data.items():
                scores = data["scores"]
                passed = data["passed"]
                n = len(scores)
                agg[model_name][metric_name] = {
                    "mean_score": round(sum(scores) / n, 4) if n else 0.0,
                    "pass_rate": round(sum(passed) / n, 4) if n else 0.0,
                    "sample_count": n,
                }

        return agg

    def _all_metrics(self, agg: dict):
        """Return a stable-ordered list of all metric names seen across all models."""
        seen = {}
        for model_stats in agg.values():
            for metric in model_stats:
                seen[metric] = True
        return list(seen.keys())

    # ------------------------------------------------------------------
    # Public outputs
    # ------------------------------------------------------------------

    def save_comparison(self):
        """Write comparison_summary.csv and comparison_detailed.json."""
        agg = self._aggregate()
        metrics = self._all_metrics(agg)

        # --- comparison_summary.csv ---
        summary_path = self.run_dir / "comparison_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_name", "metric_name", "mean_score", "pass_rate", "sample_count"],
            )
            writer.writeheader()
            for model_name in self.model_names:
                model_stats = agg.get(model_name, {})
                for metric_name in metrics:
                    stats = model_stats.get(metric_name, {"mean_score": None, "pass_rate": None, "sample_count": 0})
                    writer.writerow(
                        {
                            "model_name": model_name,
                            "metric_name": metric_name,
                            "mean_score": stats["mean_score"],
                            "pass_rate": stats["pass_rate"],
                            "sample_count": stats["sample_count"],
                        }
                    )

        # --- comparison_detailed.json ---
        # Build per-sample view: [{sample_idx, prompt, expected, models: {model_name: {response, latency_ms, evaluations}}}]
        sample_map = {}  # sample_idx -> record
        for model_name, results in self._model_results.items():
            for record in results:
                idx = record["sample_idx"]
                if idx not in sample_map:
                    sample_map[idx] = {
                        "sample_idx": idx,
                        "prompt": record["prompt"],
                        "expected": record["expected"],
                        "models": {},
                    }
                sample_map[idx]["models"][model_name] = {
                    "response": record.get("model_response", ""),
                    "latency_ms": record.get("latency_ms"),
                    "evaluations": record.get("evaluations", []),
                }

        detailed_path = self.run_dir / "comparison_detailed.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(
                [sample_map[k] for k in sorted(sample_map.keys())],
                f,
                indent=2,
                default=str,
            )

        print(f"\n  Comparison outputs saved to: {self.run_dir}")
        print(f"    {summary_path.name}")
        print(f"    {detailed_path.name}")
        return summary_path, detailed_path

    def print_comparison(self):
        """Print a formatted side-by-side table of model scores per metric."""
        agg = self._aggregate()
        metrics = self._all_metrics(agg)

        if not metrics:
            print("No comparison data to display.")
            return

        # Column widths
        metric_col = max(20, max((len(m) for m in metrics), default=0) + 2)
        model_col = 12  # minimum per-model column width
        model_cols = {m: max(model_col, len(m) + 2) for m in self.model_names}

        header_metric = f"{'Metric':<{metric_col}}"
        header_models = "".join(f"{name:>{model_cols[name]}}" for name in self.model_names)
        header = header_metric + header_models
        divider = "-" * len(header)

        print()
        print("  Model Comparison")
        print(f"  {divider}")
        print(f"  {header}")
        print(f"  {divider}")

        for metric in metrics:
            row_metric = f"{metric:<{metric_col}}"
            row_scores = ""
            for model_name in self.model_names:
                stats = agg.get(model_name, {}).get(metric)
                if stats is not None:
                    cell = f"{stats['mean_score']:.4f}"
                else:
                    cell = "N/A"
                row_scores += f"{cell:>{model_cols[model_name]}}"
            print(f"  {row_metric}{row_scores}")

        print(f"  {divider}")
        print()
