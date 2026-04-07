import csv
import json
from datetime import datetime
from pathlib import Path


class EvalReporter:
    """
    Handles saving per-sample traces, summary CSVs, and printing results
    for a single evaluation run.
    """

    def __init__(self, output_dir: str, run_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._results = []

    def record_result(
        self,
        sample_idx: int,
        prompt: str,
        expected: str,
        model_response_text: str,
        latency_ms: float,
        eval_results: list,
    ):
        """Append a per-sample result record."""
        self._results.append(
            {
                "sample_idx": sample_idx,
                "prompt": prompt,
                "expected": expected,
                "model_response": model_response_text,
                "latency_ms": latency_ms,
                "evaluations": [
                    {
                        "metric_name": r.metric_name,
                        "score": r.score,
                        "passed": r.passed,
                        "details": r.details,
                    }
                    for r in eval_results
                ],
            }
        )

    def save_detailed_results(self):
        """Write per-sample JSON trace to results_detailed.json."""
        out_path = self.run_dir / "results_detailed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, default=str)
        return out_path

    def save_summary(self):
        """
        Write summary CSV with per-evaluator stats:
        metric_name, mean_score, pass_rate, sample_count
        """
        # Aggregate per-metric
        metric_data: dict[str, dict] = {}
        for record in self._results:
            for ev in record["evaluations"]:
                name = ev["metric_name"]
                if name not in metric_data:
                    metric_data[name] = {"scores": [], "passed": []}
                metric_data[name]["scores"].append(ev["score"])
                metric_data[name]["passed"].append(ev["passed"])

        out_path = self.run_dir / "results_summary.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["metric_name", "mean_score", "pass_rate", "sample_count"],
            )
            writer.writeheader()
            for metric_name, data in metric_data.items():
                scores = data["scores"]
                passed = data["passed"]
                sample_count = len(scores)
                mean_score = sum(scores) / sample_count if sample_count else 0.0
                pass_rate = sum(passed) / sample_count if sample_count else 0.0
                writer.writerow(
                    {
                        "metric_name": metric_name,
                        "mean_score": round(mean_score, 4),
                        "pass_rate": round(pass_rate, 4),
                        "sample_count": sample_count,
                    }
                )

        return out_path

    def print_summary(self):
        """Print a clean summary table to stdout."""
        # Aggregate per-metric
        metric_data: dict[str, dict] = {}
        for record in self._results:
            for ev in record["evaluations"]:
                name = ev["metric_name"]
                if name not in metric_data:
                    metric_data[name] = {"scores": [], "passed": []}
                metric_data[name]["scores"].append(ev["score"])
                metric_data[name]["passed"].append(ev["passed"])

        if not metric_data:
            print("No evaluation results recorded.")
            return

        col_widths = {"metric": 20, "mean_score": 12, "pass_rate": 12, "samples": 10}
        header = (
            f"{'Metric':<{col_widths['metric']}}"
            f"{'Mean Score':>{col_widths['mean_score']}}"
            f"{'Pass Rate':>{col_widths['pass_rate']}}"
            f"{'Samples':>{col_widths['samples']}}"
        )
        divider = "-" * (sum(col_widths.values()))

        print()
        print("  Evaluation Summary")
        print(f"  {divider}")
        print(f"  {header}")
        print(f"  {divider}")

        for metric_name, data in metric_data.items():
            scores = data["scores"]
            passed = data["passed"]
            n = len(scores)
            mean_score = sum(scores) / n if n else 0.0
            pass_rate = sum(passed) / n if n else 0.0
            row = (
                f"{metric_name:<{col_widths['metric']}}"
                f"{mean_score:>{col_widths['mean_score']}.4f}"
                f"{pass_rate:>{col_widths['pass_rate']}.1%}"
                f"{n:>{col_widths['samples']}}"
            )
            print(f"  {row}")

        print(f"  {divider}")
        total_samples = len(self._results)
        print(f"  Total samples evaluated: {total_samples}")
        print(f"  Results saved to: {self.run_dir}")
        print()
