"""Sample plugin used by the plugin API tests. It is not installed as an entry point."""

from __future__ import annotations

from pathlib import Path


class SamplePlugin:
    name = "sample_probe"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Sample panel"

    def run(self, ctx, config):  # noqa: ARG002 - the suite runner shape
        return {
            "name": self.name,
            "source": "plugin",
            "label": "Sample plugin",
            "notes": "sample",
            "items": [],
            "done": 0,
            "total": 0,
        }

    def write(self, output_dir, summary=None):  # noqa: ARG002
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "sample-plugin.txt"
        target.write_text("sample\n", encoding="utf-8")
        return str(target)

    def wheels(self):
        return []


class FuturePlugin:
    name = "future_probe"
    version = "9.0.0"
    api_version = "1.0.0"
    title = "Future"


sample = SamplePlugin()
future = FuturePlugin()
