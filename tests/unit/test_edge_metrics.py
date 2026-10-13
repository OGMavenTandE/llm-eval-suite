"""Edge performance metrics. Readers and HTTP streams are mocked. No GPU and no network."""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval.edge.hardware import (
    cuda_from_banner,
    endpoint_is_local,
    gpu_counter_reason,
    hardware_stamp,
    parse_nvidia_smi_query,
    probe_gpu,
    ram_mib_from_bytes,
)
from llm_eval.edge.metrics import (
    format_metric,
    joules_from_power,
    percentile,
    sample_from_response,
    scorecard_edge,
    summarize_power,
    summarize_requests,
    tokens_per_second,
)
from llm_eval.edge.power import sampler_for_connection
from llm_eval.edge.session import EdgeSession
from llm_eval.edge.streaming import consume_ollama_lines, consume_openai_sse, ollama_stream_generate
from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval_suite.runs import RunManager


class StubModel(BaseModel):
    def __init__(self, reply: str = "Paris", latency_ms: float = 4.0):
        super().__init__("stub-model", {})
        self.reply = reply
        self.latency_ms = latency_ms

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(text=self.reply, latency_ms=self.latency_ms, tokens_used=2, metadata={})


class StreamModel(BaseModel):
    def __init__(self):
        super().__init__("stream-model", {})
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text="pong",
            latency_ms=40.0,
            tokens_used=4,
            metadata={"ttft_ms": 10.0, "output_tokens": 4, "eval_seconds": 0.2, "completion_tokens": 4},
        )


class _QuietSampler:
    def __init__(self, rows, *, gpu_enabled=True, gpu_reason=""):
        self.rows = rows
        self.gpu_enabled = gpu_enabled
        self.gpu_reason = gpu_reason
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        return list(self.rows)

    def summary(self, output_tokens: int) -> dict:
        return summarize_power(
            self.rows,
            gpu_enabled=self.gpu_enabled,
            gpu_reason=self.gpu_reason,
            output_tokens=output_tokens,
        )


def _stamp(**kwargs):
    defaults = dict(cpu="Test CPU", ram_total=16384.0, os_name="Windows-11", power_scheme="")
    defaults.update(kwargs)
    return hardware_stamp(**defaults)


def test_percentiles_match_the_latency_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == 38.5
    assert percentile([7.0], 95) == 7.0


def test_tokens_per_second_prefers_eval_duration_then_ttft_then_wall():
    assert tokens_per_second([{"output_tokens": 10, "eval_seconds": 0.5, "latency_ms": 1000, "ttft_ms": 100}])[0] == 20.0
    rate, basis = tokens_per_second([{"output_tokens": 10, "latency_ms": 500, "ttft_ms": 100}])
    assert basis == "time after first token"
    assert round(rate, 2) == 25.0
    rate, basis = tokens_per_second([{"output_tokens": 8, "latency_ms": 400}])
    assert basis == "full request time"
    assert rate == 20.0
    assert tokens_per_second([{"latency_ms": 10}])[1] == "no output tokens"


def test_joules_use_a_trapezoid_and_need_two_samples():
    assert joules_from_power([(0.0, 100.0)]) is None
    assert joules_from_power([(0.0, 100.0), (2.0, 200.0)]) == 300.0


def test_power_summary_degrades_without_a_gpu_or_without_tokens():
    rows = [
        {"t": 0.0, "watts": 100.0, "vram_mib": 1000.0, "ram_mib": 4000.0},
        {"t": 1.0, "watts": 200.0, "vram_mib": 1500.0, "ram_mib": 4500.0},
    ]
    remote = summarize_power(rows, gpu_enabled=False, gpu_reason="remote server; GPU counters are not on this host", output_tokens=10)
    assert remote["peak_vram_mib"]["value"] == "n/a"
    assert "remote server" in remote["peak_vram_mib"]["reason"]
    assert remote["power_watts_average"]["value"] == "n/a"
    assert remote["peak_ram_mib"]["value"] == 4500.0
    local = summarize_power(rows, gpu_enabled=True, gpu_reason="", output_tokens=10)
    assert local["power_watts_average"]["value"] == 150.0
    assert local["power_watts_peak"]["value"] == 200.0
    assert local["peak_vram_mib"]["value"] == 1500.0
    assert local["joules"]["value"] == 150.0
    assert local["joules_per_output_token"]["value"] == 15.0
    no_tokens = summarize_power(rows, gpu_enabled=True, gpu_reason="", output_tokens=0)
    assert no_tokens["joules_per_output_token"]["value"] == "n/a"
    assert "output token" in no_tokens["joules_per_output_token"]["reason"]


def test_request_summary_marks_missing_ttft():
    samples = [
        sample_from_response(ModelResponse("a", 10.0, 1, {}), 10.0),
        sample_from_response(ModelResponse("b", 30.0, 1, {"ttft_ms": 12.0, "output_tokens": 4, "eval_seconds": 0.1}), 30.0),
    ]
    summary = summarize_requests(samples, concurrency=1, source="evaluation")
    assert summary["latency_ms"]["p50"]["value"] == 20.0
    assert summary["ttft_ms"]["n"] == 1
    assert summary["ttft_ms"]["p50"]["value"] == 12.0
    assert summary["output_tokens_per_second"]["value"] == 40.0
    assert "connector did not stream" in samples[0]["ttft_reason"]


def test_a_stream_with_no_token_is_not_a_missing_connector():
    empty = sample_from_response(
        ModelResponse(
            text="",
            latency_ms=12.0,
            tokens_used=0,
            metadata={"streaming": True, "ttft_ms": None, "completion_tokens": 0},
        )
    )
    assert empty["ttft_reason"] == "stream returned no token"
    assert empty["ttft_ms"] is None
    silent = sample_from_response(ModelResponse(text="pong", latency_ms=12.0, tokens_used=1, metadata={}))
    assert silent["ttft_reason"] == "connector did not stream"
    streamed = sample_from_response(
        ModelResponse(
            text="pong",
            latency_ms=40.0,
            tokens_used=1,
            metadata={"streaming": True, "ttft_ms": 8.5, "completion_tokens": 1},
        )
    )
    assert streamed["ttft_ms"] == 8.5
    assert streamed["ttft_reason"] == ""
    assert streamed["output_tokens"] == 1


def test_ollama_metadata_counts_as_output_tokens():
    sample = sample_from_response(
        ModelResponse("a", 100.0, 8, {"prompt_eval_count": 3, "eval_duration_ns": 250_000_000}),
        100.0,
    )
    assert sample["output_tokens"] == 8
    assert sample["eval_seconds"] == 0.25


def test_hardware_stamp_and_remote_gpu_counters():
    total, used = ram_mib_from_bytes(8 * 1048576, 2 * 1048576)
    assert (total, used) == (8.0, 6.0)
    assert cuda_from_banner("CUDA Version: 12.6") == "12.6"
    parsed = parse_nvidia_smi_query("NVIDIA GeForce RTX 4070 Super, 560.94, 4096, [N/A], 220")
    assert parsed["name"] == "NVIDIA GeForce RTX 4070 Super"
    assert parsed["watts"] is None
    assert parsed["power_limit_w"] == 220.0
    assert parse_nvidia_smi_query("") is None

    local = _stamp(
        connection={"type": "ollama", "base_url": "http://127.0.0.1:11434"},
        probe=lambda: {
            "name": "NVIDIA GeForce RTX 4070 Super",
            "driver": "560.94",
            "cuda": "12.6",
            "power_limit_w": 220,
        },
        power_scheme="High performance",
    )
    assert local["gpu"]["value"] == "NVIDIA GeForce RTX 4070 Super"
    assert local["driver"]["value"] == "560.94"
    assert local["cuda"]["value"] == "12.6"
    assert local["cpu"]["value"] == "Test CPU"
    assert local["ram_mib"]["value"] == 16384.0
    assert local["os"]["value"] == "Windows-11"
    assert local["power_mode"]["value"] == "220 W limit, High performance"
    assert local["endpoint"] == "local"
    assert endpoint_is_local({"type": "hf", "folder": "C:/models/q4"}) is True

    missing = _stamp(connection={"type": "ollama", "base_url": "http://127.0.0.1:11434"}, probe=lambda: None)
    assert missing["gpu"]["value"] == "n/a"
    assert missing["gpu"]["reason"] == "no NVIDIA GPU detected"
    assert missing["cuda"]["value"] == "n/a"
    assert missing["power_mode"]["value"] == "n/a"
    assert format_metric(missing["driver"]) == "n/a (no NVIDIA GPU detected)"

    remote = _stamp(
        connection={"type": "openai", "base_url": "https://api.openai.com/v1", "cloud": True},
        probe=lambda: {"name": "A100", "driver": "550", "cuda": "12.4", "power_limit_w": 300},
    )
    assert remote["endpoint"] == "remote"
    assert gpu_counter_reason(remote and {"cloud": True, "base_url": "https://api.openai.com/v1"}, True).startswith("remote")
    sampler = sampler_for_connection(
        {"type": "openai", "base_url": "https://api.openai.com/v1", "cloud": True},
        probe=lambda: {"name": "A100", "watts": 250, "vram_mib": 40000},
    )
    assert sampler.gpu_enabled is False
    assert "remote server" in sampler.gpu_reason
    bare = sampler_for_connection({"type": "ollama", "base_url": "http://127.0.0.1:11434"}, probe=lambda: None)
    assert bare.gpu_enabled is False
    assert bare.gpu_reason == "no NVIDIA GPU detected"


def test_nvml_is_preferred_and_smi_is_the_fallback():
    calls = {"smi": 0}

    def smi():
        calls["smi"] += 1
        return {"name": "from-smi", "driver": "1", "cuda": "12.1", "watts": 10, "vram_mib": 20, "power_limit_w": 30}

    preferred = probe_gpu(
        nvml_reader=lambda: {"name": "RTX 4070 Super", "driver": "560", "cuda": "12.6", "watts": 180, "vram_mib": 9000, "power_limit_w": 220},
        smi_reader=smi,
    )
    assert preferred["name"] == "RTX 4070 Super"
    assert calls["smi"] == 0
    fallback = probe_gpu(nvml_reader=lambda: None, smi_reader=smi)
    assert fallback["name"] == "from-smi"
    filled = probe_gpu(
        nvml_reader=lambda: {"name": "RTX", "driver": "560", "cuda": "", "watts": 1, "vram_mib": 2, "power_limit_w": None},
        smi_reader=smi,
    )
    assert filled["cuda"] == "12.1"
    assert filled["power_limit_w"] == 30


def test_stream_parsers_take_ttft_from_the_first_token():
    clock = iter([10.05, 10.20]).__next__
    ttft, text, last, latency = consume_ollama_lines(
        [b'{"response":""}', b'{"response":"Po","done":false}', b'{"response":"ng","done":true,"eval_count":2}'],
        10.0,
        clock=clock,
    )
    assert round(ttft, 6) == 50.0
    assert text == "Pong"
    assert last["eval_count"] == 2
    assert round(latency, 6) == 200.0

    times = iter([5.02, 5.05])

    def sse_clock():
        return next(times)
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "Pong"}, "finish_reason": "stop"}]}),
        "data: " + json.dumps({"usage": {"completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4}}),
        "data: [DONE]",
    ]
    ttft, text, usage, finish, _model, _latency = consume_openai_sse(lines, 5.0, "chat", clock=sse_clock)
    assert text == "Pong"
    assert round(ttft, 6) == 20.0
    assert usage["completion_tokens"] == 1
    assert finish == "stop"


def test_ollama_stream_hook_records_ttft_and_falls_back(monkeypatch):
    class Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield b'{"response":""}'
            yield b'{"response":"Pong","done":true,"eval_count":1,"eval_duration":200000000,"prompt_eval_count":3,"model":"llama"}'

    monkeypatch.setattr("llm_eval.edge.streaming.requests.post", lambda *args, **kwargs: Resp())
    result = ollama_stream_generate("http://127.0.0.1:11434", {"model": "llama", "prompt": "hi"})
    assert result.text == "Pong"
    assert result.metadata["ttft_ms"] is not None
    assert result.metadata["output_tokens"] == 1
    assert result.metadata["eval_seconds"] == 0.2
    assert result.metadata["stream"] is True

    calls = {"n": 0}

    class JsonResp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "Pong", "eval_count": 1, "prompt_eval_count": 2, "model": "llama"}

    def post(url, json=None, timeout=None, stream=None):  # noqa: A002
        calls["n"] += 1
        if stream:
            raise ValueError("bad stream")
        return JsonResp()

    monkeypatch.setattr("llm_eval.edge.streaming.requests.post", post)
    fallback = ollama_stream_generate("http://127.0.0.1:11434", {"model": "llama", "prompt": "hi"})
    assert calls["n"] == 2
    assert fallback.metadata["stream"] is False
    assert fallback.metadata["ttft_reason"].startswith("streaming failed")
    assert fallback.metadata["output_tokens"] == 1


def test_session_load_probe_and_scorecard_block():
    rows = [
        {"t": 0.0, "watts": 100.0, "vram_mib": 8000.0, "ram_mib": 12000.0},
        {"t": 1.0, "watts": 140.0, "vram_mib": 9000.0, "ram_mib": 13000.0},
    ]
    hardware = _stamp(
        connection={"type": "ollama", "base_url": "http://127.0.0.1:11434"},
        probe=lambda: {"name": "RTX 4070 Super", "driver": "560", "cuda": "12.6", "power_limit_w": 220},
    )
    model = StreamModel()
    session = EdgeSession(
        {"type": "ollama"},
        concurrency=4,
        sampler=_QuietSampler(rows),
        hardware=hardware,
    )
    session.wrap(model)
    session.start()
    session.probe()
    summary = session.finish()
    assert model.calls == 4
    assert summary["concurrency"] == 4
    assert summary["source"] == "load_probe"
    assert summary["requests"] == 4
    assert summary["ttft_ms"]["p50"]["value"] == 10.0
    assert summary["latency_ms"]["p95"]["value"] == 40.0
    assert summary["output_tokens_per_second"]["value"] == 20.0
    assert summary["joules"]["value"] == 120.0
    assert summary["joules_per_output_token"]["value"] == 7.5
    assert summary["serial"]["requests"] == 0
    card = scorecard_edge(summary)
    assert card["hardware"]["gpu"]["value"] == "RTX 4070 Super"
    assert card["peak_vram_mib"]["value"] == 9000.0
    assert session.finish() is summary


def test_run_records_edge_metrics_and_keeps_them_when_invalid(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("llm_eval.garak.live.garak_is_installed", lambda: False)
    dataset = tmp_path / "facts.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"prompt": "Capital of France?", "expected_answer": "Paris"}) + "\n"
            for _ in range(2)
        ),
        encoding="utf-8",
    )
    manager = RunManager(tmp_path / "runs", model_factory=lambda _profile: StubModel("Paris", latency_ms=4.0))
    record = manager.start(
        connection={"type": "openai", "model": "stub-model", "base_url": "http://127.0.0.1:9/v1"},
        preset_id="quick",
        dataset_path=str(dataset),
        background=False,
    )
    assert record["status"] == "completed"
    assert record["hardware"]["cpu"]["value"]
    assert record["hardware"]["os"]["value"]
    edge = record["edge_metrics"]
    assert edge["concurrency"] == 1
    assert edge["latency_ms"]["p50"]["value"] == 4.0
    assert edge["latency_ms"]["p95"]["value"] == 4.0
    assert edge["ttft_ms"]["value"] == "n/a"
    assert "stream" in edge["ttft_ms"]["reason"]
    assert record["scorecard"]["edge"]["latency_ms"]["p50"]["value"] == 4.0
    vram = edge["peak_vram_mib"]
    if vram["value"] == "n/a":
        assert vram["reason"]
    html = manager.report_html(record["run_id"])
    assert "Edge performance" in html
    assert "n/a" in html

    empty = RunManager(tmp_path / "empty-runs", model_factory=lambda _profile: StubModel("", latency_ms=3.0))
    invalid = empty.start(
        connection={"type": "openai", "model": "stub-model", "base_url": "http://127.0.0.1:9/v1"},
        preset_id="quick",
        dataset_path=str(dataset),
        background=False,
    )
    assert invalid["status"] == "invalid"
    assert invalid["scorecard"]["categories"]
    assert any(row["status"] == "withheld" for row in invalid["scorecard"]["categories"])
    assert invalid["scorecard"]["edge"]["latency_ms"]["p50"]["value"] == 3.0
    assert invalid["edge_metrics"]["peak_ram_mib"]["value"] != "n/a" or invalid["edge_metrics"]["peak_ram_mib"]["reason"]
