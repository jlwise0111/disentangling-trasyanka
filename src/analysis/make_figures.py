"""
Figures for the presentation, from the saved results (no training). Writes PNGs to
results/figures/, sized for a 16:9 slide:

    1. main_results.png      Trasyanka test Macro-F1 for every model in zero-shot, in-domain, mixed
    2. learning_curve.png    from 0 % to 100 % of the Trasyanka training data (mixed setting):
                             left BE / RU / SHARED (3-class Macro-F1), right HYBRID F1
    3. stress_test.png       from written Russian and Belarusian to speech, zero-shot and mixed
    4. hybrid_misses.png     gold HYBRID words by how many models miss them, split by whether
                             the word form was seen in training

Colours: one fixed colour and marker per model in every figure (a validated colour-blind-safe
categorical order), so CRF is always blue circles, BiLSTM orange squares, XLM-R aqua diamonds,
SlavicBERT yellow triangles. Lines are labelled directly at their right end as well.

Run from the project root:   python src/analysis/make_figures.py
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # no window: write files only
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.baseline.splits import RESULTS_DIR
from src.analysis.compare_models import run_name
from src.analysis.stress_test import MODELS, load_results, report_scores, three_class_macro_f1, stress_row

FIGURES_DIR = RESULTS_DIR / "figures"

# Share of the Trasyanka training transcripts -> the run that used it. 0 % is zero-shot
# (treebanks only); the others are the mixed setting (treebanks + that share of Trasyanka).
SHARE_RUNS = {0: ("zero_shot", 1.0), 10: ("mixed", 0.1), 30: ("mixed", 0.3),
              70: ("mixed", 0.7), 100: ("mixed", 1.0)}
STRESS_SETS = ["Russian\n(written)", "Belarusian\n(official)", "Taraškievica\n(other spelling)",
               "Trasyanka\n(speech, 3-class)"]
STRESS_KEYS = ["ru_test", "be_official", "taraskievica", "trasyanka_3"]   # stress_row keys, same order

# Validated categorical order (blue, orange, aqua, yellow), one per model, plus a marker each
# so identity never depends on colour alone
COLORS = {"CRF": "#2a78d6", "BiLSTM": "#eb6834", "XLM-R": "#1baf7a", "SlavicBERT": "#eda100"}
MARKERS = {"CRF": "o", "BiLSTM": "s", "XLM-R": "D", "SlavicBERT": "^"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def learning_curve(prefix, metric):
    """
    One value per share in SHARE_RUNS (in that order) for one model, on trasyanka_test:
        metric "four_class"   the saved macro_f1
               "three_class"  three_class_macro_f1 of the saved report
               "hybrid_f1"    the HYBRID F1 from the saved report
    None for a run whose results file is missing.
        learning_curve("xlmr", "four_class") -> [0.350, 0.620, 0.7956, 0.8622, 0.8829]
    Raises ValueError for any other metric.
    """
    # Checked before the loop, so a typo fails at once instead of after loading five files
    if metric not in ("four_class", "three_class", "hybrid_f1"):
        raise ValueError(f"unknown metric {metric!r}")
    values = []
    for setting, fraction in SHARE_RUNS.values():
        results = load_results(run_name(prefix, setting, fraction))
        if results is None:
            values.append(None)
            continue
        test = results["trasyanka_test"]
        if metric == "four_class":
            values.append(test["macro_f1"])
        elif metric == "three_class":
            values.append(three_class_macro_f1(report_scores(test["report"])))
        else:                                   # "hybrid_f1"
            values.append(report_scores(test["report"])["HYBRID"]["f1"])
    return values


def stress_values(prefix, setting):
    """
    The four stress-test scores of one model in one setting, in STRESS_SETS order:
        [RU F1 on the Russian test, BE F1 on official Belarusian, BE F1 on Taraškievica,
         Trasyanka 3-class Macro-F1]
    from stress_row() of that run (fraction 1.0). None if the run is missing.
    """
    results = load_results(run_name(prefix, setting, 1.0))
    if results is None:
        return None
    row = stress_row(results)
    return [row[key] for key in STRESS_KEYS]



# ----------------------------------------------------------------------------- plotting


def style():
    """Quiet chart chrome: light surface, hairline y grid, no top/right frame, system sans."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 12,
    })


def legend_on_top(fig, models):
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linewidth=2, markersize=8) for m in models]
    fig.legend(handles, models, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 0.995))


def end_labels(ax, x, series, gap_share=0.05):
    """
    Model names at the right end of each line, nudged apart so they never overlap. The gap is a
    share of the panel's y-range (set the y-limits first); runs of labels that had to be pushed
    apart are re-centred on their points, so no label drifts far from its line.
    """
    low, high = ax.get_ylim()
    gap = gap_share * (high - low)
    ends = sorted(((values[-1], model) for model, values in series.items() if values[-1] is not None))
    # Group labels that are closer than the gap, spread each group evenly around its mean
    groups = []
    for y, model in ends:
        if groups and y - groups[-1][-1][0] < gap:
            groups[-1].append((y, model))
        else:
            groups.append([(y, model)])
    placed = []
    for group in groups:
        centre = sum(y for y, _ in group) / len(group)
        start = centre - gap * (len(group) - 1) / 2
        if placed and start < placed[-1] + gap:                  # never overlap the group below
            start = placed[-1] + gap
        for k, (_, model) in enumerate(group):
            y = start + k * gap
            placed.append(y)
            ax.annotate(model, (x, y), xytext=(10, 0), textcoords="offset points", va="center",
                        color=INK_2, fontsize=11)


def main_results(path):
    settings = [("zero_shot", "Zero-shot\n(treebanks only)"), ("in_domain", "In-domain\n(Trasyanka only)"),
                ("mixed", "Mixed\n(both)")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    width = 0.19
    for i, (model, prefix) in enumerate(MODELS.items()):
        values = [load_results(run_name(prefix, s, 1.0))["trasyanka_test"]["macro_f1"] for s, _ in settings]
        xs = [k + (i - 1.5) * width for k in range(len(settings))]
        # edgecolor = surface: a 2px gap between neighbouring bars
        ax.bar(xs, values, width, color=COLORS[model], edgecolor=SURFACE, linewidth=2, label=model)
    # Label only the best bar of each setting
    for k, (setting, _) in enumerate(settings):
        scores = {m: load_results(run_name(p, setting, 1.0))["trasyanka_test"]["macro_f1"] for m, p in MODELS.items()}
        best = max(scores, key=scores.get)
        i = list(MODELS).index(best)
        ax.annotate(f"{scores[best]:.3f}", (k + (i - 1.5) * width, scores[best]), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK, fontsize=12, fontweight="semibold")
    ax.set_xticks(range(len(settings)), [label for _, label in settings])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (BE, RU, SHARED, HYBRID)")
    ax.tick_params(axis="x", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODELS]
    fig.legend(handles, list(MODELS), loc="upper center", ncol=len(MODELS), bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def learning_curve_figure(path):
    shares = list(SHARE_RUNS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True)
    panels = [("three_class", "BE, RU, SHARED  (3-class Macro-F1)", (0.4, 1.0)),
              ("hybrid_f1", "HYBRID F1", (0.0, 0.85))]
    for ax, (metric, title, ylim) in zip(axes, panels):
        series = {model: learning_curve(prefix, metric) for model, prefix in MODELS.items()}
        for model, values in series.items():
            ax.plot(shares, values, color=COLORS[model], marker=MARKERS[model], linewidth=2, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)   # surface ring where markers overlap
        ax.set_ylim(*ylim)                                           # before end_labels, which uses it
        end_labels(ax, shares[-1], series)
        ax.set_title(title, loc="left")
        ax.set_xticks(shares, [f"{s} %" for s in shares])
        ax.set_xlim(-4, 118)                                         # room for the end labels
        ax.set_xlabel("Share of the Trasyanka training transcripts (0 % = zero-shot)")
    legend_on_top(fig, list(MODELS))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stress_figure(path):
    xs = list(range(len(STRESS_SETS)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    axes[0].set_ylim(0.4, 1.02)                                      # shared; before end_labels
    for ax, (setting, title) in zip(axes, [("zero_shot", "Zero-shot (treebanks only)"),
                                           ("mixed", "Mixed (treebanks + Trasyanka)")]):
        series = {model: stress_values(prefix, setting) for model, prefix in MODELS.items()}
        for model, values in series.items():
            ax.plot(xs, values, color=COLORS[model], marker=MARKERS[model], linewidth=2, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        end_labels(ax, xs[-1], series)
        ax.set_title(title, loc="left")
        ax.set_xticks(xs, STRESS_SETS, fontsize=11)
        ax.set_xlim(-0.3, len(xs) - 1 + 0.75)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("F1 of the set's language / Macro-F1")
    legend_on_top(fig, list(MODELS))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def hybrid_misses_figure(path):
    with open(RESULTS_DIR / "hybrid_errors.json", encoding="utf-8") as f:
        by_count = json.load(f)["train_status_by_missed_count"]
    # Unseen forms are the story, so they get the strong colour; seen forms are shades of blue / grey
    statuses = [("unseen", "form unseen in training", "#eb6834"),
                ("always HYBRID", "always HYBRID in training", "#2a78d6"),
                ("sometimes HYBRID", "sometimes HYBRID", "#86b6ef"),
                ("never HYBRID", "never HYBRID", "#c3c2b7")]
    rows = [str(k) for k in range(5)]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    left = [0] * len(rows)
    for status, label, color in statuses:
        widths = [by_count[k].get(status, 0) for k in rows]
        ax.barh(rows, widths, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for y, (x0, w) in enumerate(zip(left, widths)):
            if w >= 45:                                             # label only segments wide enough
                total = sum(by_count[rows[y]].values())
                ax.text(x0 + w / 2, y, f"{w / total:.0%}", ha="center", va="center", fontsize=11,
                        color="#ffffff" if color != "#c3c2b7" and color != "#86b6ef" else INK)
        left = [a + b for a, b in zip(left, widths)]
    ax.set_yticks(range(len(rows)), [f"missed by {k}" for k in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Gold HYBRID words in the test families (1,205)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(figures_dir=FIGURES_DIR):
    style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in [("main_results", main_results), ("learning_curve", learning_curve_figure),
                       ("stress_test", stress_figure), ("hybrid_misses", hybrid_misses_figure)]:
        path = figures_dir / f"{name}.png"
        draw(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
