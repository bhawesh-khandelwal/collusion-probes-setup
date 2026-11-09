"""Paper figures and tables from the saved result files. Output: report/figures/*.png|pdf, report/tables.md."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, SURF = "#1f1f1e", "#5b5a55", "#e4e3de", "#fcfcfb"
COL = {"peak_suspicion": "#2a78d6", "asymmetry_probe": "#eb6834", "set_stats": "#1baf7a",
       "pair_max": "#eda100", "monitor": "#6250d6"}
MARK = {"peak_suspicion": "o", "asymmetry_probe": "s", "set_stats": "D", "pair_max": "^", "monitor": "v"}
NAME = {"peak_suspicion": "Peak Suspicion (per-agent)", "asymmetry_probe": "Asymmetry Probe (per-agent)",
        "set_stats": "set_stats (group, preselected)", "pair_max": "pair_max (group)",
        "monitor": "Text monitor (public transcript)"}
TIERS = [("transfer", "Instructed\ncollusion"), ("implicit", "Implicit setup\nvs control"),
         ("implicit2_coord", "Same setup:\ncoordinated vs not")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})


def j(p):
    return json.loads((ROOT / p).read_text())


def probe(model, tier):
    if model == "27B":
        src = "results/wp5_implicit_discussion.json" if tier == "implicit2_coord" else "results/wp4c_discussion.json"
    else:
        src = "results/gemma_discussion.json"
    return j(src)[tier]["all"]


MON = j("results/monitor_eval.json")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1():
    """AUROC by test set: detection falls to near chance once the setup is held fixed."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    methods = list(COL)
    for ax, model in zip(axes, ("27B", "Gemma")):
        for ti, (tier, label) in enumerate(TIERS):
            for mi, m in enumerate(methods):
                x = ti + (mi - 2) * 0.14
                if m == "monitor":
                    a, ci = MON[tier]["auroc"], MON[tier]["ci"]
                else:
                    a, ci = probe(model, tier)[m]["auroc"], probe(model, tier)[m]["ci"]
                ax.plot([x, x], ci, color=COL[m], lw=2, solid_capstyle="round", zorder=2)
                ax.plot(x, a, MARK[m], color=COL[m], ms=6, mec=SURF, mew=1.2, zorder=3,
                        label=NAME[m] if ti == 0 else None)
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(-0.45, 0.505, "chance", color=INK2, fontsize=7, va="bottom", ha="left")
        ax.set_xticks(range(3), [t[1] for t in TIERS], fontsize=8)
        ax.set_ylim(0.2, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"Probes on {'Qwen3.8-27B' if model == '27B' else 'Gemma 4 31B'}", fontsize=9, loc="left")
    axes[0].set_ylabel("AUROC (95% CI)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.2))
    fig.text(0.0, -0.25, "Probes trained on Core (instructed collusion), tested zero-shot; discussion-turn read. "
             "Text monitor is the same in both panels.", fontsize=6.5, color=INK2)
    save(fig, "fig1_setup_vs_coordination")


SIG = set()


def parse_vs_peak(path):
    rows = {}
    for line in (ROOT / path).read_text().splitlines():
        m = re.match(r"(\S+)\s+(\S+)\s+(\S+)\s+- peak ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if m:
            rows[(m[1], m[2], m[3])] = (float(m[4]), float(m[5]), float(m[6]))
            if line.rstrip().endswith("*"):  # the log flags significance on unrounded bounds
                SIG.add((m[1], m[2], m[3]))
    return rows


def fig2(vs):
    """Group minus per-agent max, every test set, both models, both read positions."""
    items = []
    for f, model in (("gemma_scores_discussion.json", "Gemma disc."), ("gemma_scores_final.json", "Gemma final"),
                     ("scores_27b_transfer_discussion.json", "27B disc."), ("scores_27b_transfer_final.json", "27B final"),
                     ("scores_implicit_discussion.json", "27B disc."), ("scores_implicit_final.json", "27B final")):
        for (ff, tier, m), v in vs.items():
            if ff == f and tier in ("transfer", "implicit", "implicit_coord", "implicit2_coord", "game"):
                items.append((tier, model, m, v, (ff, tier, m) in SIG))
    order = {"transfer": 0, "implicit": 1, "implicit_coord": 2, "implicit2_coord": 3, "game": 4}
    items.sort(key=lambda t: (order[t[0]], t[1], t[2]))
    TN = {"transfer": "Instructed", "implicit": "Implicit vs control", "implicit_coord": "Same setup (36)",
          "implicit2_coord": "Same setup (166)", "game": "Pricing game"}
    rows = sorted({(t, mo) for t, mo, _, _, _ in items}, key=lambda r: (order[r[0]], r[1]))
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(rows) + 0.9))
    for t, mo, m, (d, lo, hi), _ in items:
        y = len(rows) - rows.index((t, mo)) + (0.17 if m == "set_stats" else -0.17)
        ax.plot([lo, hi], [y, y], color=COL[m], lw=2, solid_capstyle="round")
        ax.plot(d, y, MARK[m], color=COL[m], ms=5, mec=SURF, mew=1)
    for k in range(1, len(rows)):
        if rows[k][0] != rows[k - 1][0]:
            ax.axhline(len(rows) - k + 0.5, color=GRID, lw=0.8)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows), 0, -1), [f"{TN[t]} · {mo}" for t, mo in rows], fontsize=7)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_xlabel("Group AUROC minus Peak Suspicion AUROC (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for m in ("set_stats", "pair_max"):
        ax.plot([], [], MARK[m], color=COL[m], label=NAME[m])
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.4, 1.0))
    save(fig, "fig2_group_minus_peak")
    return items


def fig3():
    """In-domain training: held-out scenario looks good, held-out family is chance."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    meths = [("peak", "peak_suspicion", "Peak (per-agent)"), ("set_stats", "set_stats", "set_stats (group)"),
             ("pair_max", "pair_max", "pair_max (group)")]
    for ax, model in zip(axes, ("27b", "gemma")):
        d = j(f"results/indomain_{model}.json")
        for si, split in enumerate(("scenario", "family")):
            for mi, (k, c, lab) in enumerate(meths):
                x = si + (mi - 1) * 0.22
                r = d[f"{split}|{k}"]
                ax.bar(x, r["auroc"] - 0.5, bottom=0.5, width=0.18, color=COL[c],
                       label=lab if si == 0 else None, zorder=2)
                ax.plot([x, x], r["ci"], color=INK, lw=1, zorder=3)
        fp = d["scenario|family_prior"]["auroc"]
        ax.plot([-0.4, 0.4], [fp, fp], color=INK, lw=1.5, ls=(0, (4, 2)), zorder=4)
        ax.text(-0.4, fp + 0.012, "family base rate only", fontsize=6.5, va="bottom", color=INK)
        ax.axhline(0.5, color=INK2, lw=1, zorder=1)
        ax.set_xticks([0, 1], ["Scenario held out", "Family held out"])
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title("Qwen3.8-27B" if model == "27b" else "Gemma 4 31B", fontsize=9, loc="left")
    axes[0].set_ylabel("AUROC (95% CI)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig3_indomain")


def tables(items):
    lines = ["# Tables (generated by report/make_figures.py)", ""]
    lines += ["## Table 1. AUROC [95% CI], discussion read, trained on Core", "",
              "| Model | Test set | Peak | Asymmetry | set_stats | pair_max | Text monitor |", "|---|---|---|---|---|---|---|"]
    for model in ("27B", "Gemma"):
        for tier, lab in TIERS + [("implicit_coord", "Same setup (original 36)")]:
            p = probe(model, tier)
            cells = [f"{p[m]['auroc']:.2f} [{p[m]['ci'][0]:.2f}, {p[m]['ci'][1]:.2f}]"
                     for m in ("peak_suspicion", "asymmetry_probe", "set_stats", "pair_max")]
            mon = MON[tier]
            cells.append(f"{mon['auroc']:.2f} [{mon['ci'][0]:.2f}, {mon['ci'][1]:.2f}]")
            lines.append(f"| {model} | {lab.replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    lines += ["", "## Table 2. Group minus Peak Suspicion [95% CI]", "", "| Test set | Model, read | Method | Difference |",
              "|---|---|---|---|"]
    for tier, model, m, (d, lo, hi), sig in items:
        star = " *" if sig else ""
        lines.append(f"| {tier} | {model} | {m} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star} |")
    lines += ["", "## Table 3. In-domain detectors on same-setup runs (AUROC [95% CI], within-family AUROC)", "",
              "| Model | Split | Peak | set_stats | pair_max | Family base rate only |", "|---|---|---|---|---|---|"]
    for model in ("27b", "gemma"):
        d = j(f"results/indomain_{model}.json")
        for split in ("scenario", "family"):
            c = [f"{d[f'{split}|{k}']['auroc']:.2f} [{d[f'{split}|{k}']['ci'][0]:.2f}, {d[f'{split}|{k}']['ci'][1]:.2f}]; "
                 f"{d[f'{split}|{k}']['within_family']:.2f}" for k in ("peak", "set_stats", "pair_max")]
            fp = d[f"{split}|family_prior"]["auroc"]
            lines.append(f"| {model} | {split} held out | " + " | ".join(c) + f" | {fp:.2f} |")
    (ROOT / "report" / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig1()
    vs = parse_vs_peak("results/vs_peak_all.log")
    items = fig2(vs)
    fig3()
    tables(items)
    print("figures:", sorted(p.name for p in OUT.iterdir()))
