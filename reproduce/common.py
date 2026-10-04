"""Shared paths, readers and plotting style for the quick reproduction."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
OUT = REPO / 'outputs'

STOP = 'P1_continue_vs_stop'      # stop question
INCREASE = 'P2_current_vs_1p5x'   # increase question (1.5 times)
QUESTIONS = (STOP, INCREASE)
LEADS = (10, 30)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def num(value):
    """Float or None for blank/None/non-finite strings."""
    if value in ('', None, 'None', 'nan', 'NaN'):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def detected(row: dict) -> bool:
    return row['effect_detected'] == 'True'


def style():
    """Times New Roman when installed (journal figures), otherwise a serif fallback."""
    import matplotlib
    from matplotlib import font_manager
    try:
        font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
        family = 'Times New Roman'
    except ValueError:
        family = 'serif'
    matplotlib.rcParams.update({
        'font.family': family, 'mathtext.fontset': 'stix', 'axes.labelweight': 'bold', 'axes.labelsize': 13,
        'axes.titleweight': 'bold', 'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    return family


def letter(ax, text):
    ax.text(-.12, 1.03, text, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
