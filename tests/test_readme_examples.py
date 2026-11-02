"""The Python examples in README.md must run."""

import re
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_examples_run():
    if not README.exists():
        pytest.skip("README.md not present")
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    assert blocks, "no python example in README.md"
    namespace: dict = {}
    for block in blocks:  # blocks share a namespace, in order
        exec(compile(block, str(README), "exec"), namespace)
