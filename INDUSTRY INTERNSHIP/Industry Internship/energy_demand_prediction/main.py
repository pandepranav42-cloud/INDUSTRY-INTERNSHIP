"""Run the full reproduction of Kamoona et al. (2023):

    python main.py

Outputs are written to ./outputs (figures, tables, generated data and a
plain-text results summary).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_dataset1 import run as run_ds1   # noqa: E402
from run_dataset2 import run as run_ds2   # noqa: E402


def main() -> None:
    out = ROOT / "outputs"
    lines: list[str] = []
    lines += run_ds1(out)
    lines.append("")
    lines += run_ds2(out)

    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"\nFigures -> {out / 'figures'}")
    print(f"Tables  -> {out / 'tables'}")
    print(f"Data    -> {out / 'data'}")


if __name__ == "__main__":
    main()
