"""Stage 2 figures. Same validated categorical palette and quiet styling as plots.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .plots import GRID, INK, MUTED, SERIES, SURFACE, _style  # noqa: E402


def _panels(n, w, h, sharey=True):
    fig, axes = plt.subplots(1, n, figsize=(w, h), dpi=150, sharey=sharey)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


def altitude_vs_wall(rows, path, rule=0.10):
    """rows: dicts with layout, partition_h, clutter_h, gain=(mean, lo, hi). One panel per layout."""
    layouts = [l for l in ("open", "cubicle", "booth") if any(r["layout"] == l for r in rows)]
    fig, axes = _panels(len(layouts), 3.6 * len(layouts) + 0.8, 3.9)
    for ax, layout in zip(axes, layouts):
        for i, cl in enumerate(sorted({r["clutter_h"] for r in rows if r["layout"] == layout})):
            sel = sorted([r for r in rows if r["layout"] == layout and r["clutter_h"] == cl],
                         key=lambda r: r["partition_h"])
            x = np.array([r["partition_h"] for r in sel])
            m = np.array([r["gain"][0] for r in sel]) * 100
            lo = np.array([r["gain"][1] for r in sel]) * 100
            hi = np.array([r["gain"][2] for r in sel]) * 100
            ax.errorbar(x + (i - 0.5) * 0.02, m, yerr=[m - lo, hi - m], color=SERIES[i], linewidth=2, marker="o",
                        markersize=5, capsize=3, label=("no desk clutter" if cl == 0 else f"clutter {cl:g} m"))
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.axhline(rule * 100, color=SERIES[3], linewidth=1.2, linestyle="--")
        ax.set_title(f"{layout} layout", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Partition / cubicle wall height (m)", color=INK, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Time saved by using all altitudes\nvs flying only at 0.4 m (%)", color=INK, fontsize=9)
    axes[-1].text(0.98, rule * 100 + 0.3, "10 % rule", color=SERIES[3], fontsize=8, ha="right",
                  transform=axes[-1].get_yaxis_transform())
    fig.suptitle("Does altitude pay off? Time saving across layouts (mean and 95 % interval over seeds)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def seed_strips(data, path):
    """data: {metric_label: {layout: array over seeds (fractions)}}."""
    metrics = list(data)
    fig, axes = _panels(len(metrics), 4.3 * len(metrics), 3.9, sharey=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, metrics):
        layouts = list(data[metric])
        for i, lay in enumerate(layouts):
            v = np.asarray(data[metric][lay]) * 100
            ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=SERIES[i % 4], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=INK, linewidth=2.2)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(layouts)))
        ax.set_xticklabels(layouts, fontsize=8, rotation=12)
        ax.set_title(metric, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("%", color=INK, fontsize=9)
    fig.suptitle("Seed-to-seed spread (each dot is one simulated six-week world; bar = mean)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sensitivity_dots(per_base, path, rule=0.10):
    """per_base: {base_label: [(setting_label, (m, lo, hi)), ...]} altitude saving."""
    fig, axes = plt.subplots(1, len(per_base), figsize=(5.6 * len(per_base), 6.4), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (base, items) in zip(axes, per_base.items()):
        _style(ax)
        ys = np.arange(len(items))[::-1]
        for y, (label, (m, lo, hi)) in zip(ys, items):
            ax.errorbar(m * 100, y, xerr=[[(m - lo) * 100], [(hi - m) * 100]], fmt="o", color=SERIES[0],
                        markersize=5, capsize=3, linewidth=1.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([l for l, _ in items], fontsize=8)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.axvline(rule * 100, color=SERIES[3], linewidth=1.2, linestyle="--")
        ax.set_title(base, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Time saved by altitude (%), 95 % interval over seeds", color=INK, fontsize=9)
        ax.grid(True, axis="x", color=GRID)
        ax.grid(False, axis="y")
    fig.suptitle("Sensitivity of the altitude benefit to drone and sensing assumptions (dashed = 10 % rule)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lookalike_lines(groups, path):
    """groups: {base: {group_title: [(x_label, {arm: (m, lo, hi)}), ...]}}."""
    bases = list(groups)
    titles = list(next(iter(groups.values())))
    fig, axes = plt.subplots(len(bases), len(titles), figsize=(4.2 * len(titles), 3.5 * len(bases)), dpi=150,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    arms = ["trust", "trust (no spatial re-ID)", "ours", "ours (no spatial re-ID)"]
    for r, base in enumerate(bases):
        for c, title in enumerate(titles):
            ax = axes[r][c]
            _style(ax)
            items = groups[base][title]
            for i, arm in enumerate(arms):
                m = np.array([it[1][arm][0] for it in items])
                lo = np.array([it[1][arm][1] for it in items])
                hi = np.array([it[1][arm][2] for it in items])
                x = np.arange(len(items)) + (i - 1.5) * 0.06
                ax.errorbar(x, m, yerr=[m - lo, hi - m], color=SERIES[i], marker="o", markersize=4, linewidth=1.8,
                            capsize=2, label=arm, linestyle="--" if "no spatial" in arm else "-")
            ax.set_xticks(range(len(items)))
            ax.set_xticklabels([it[0] for it in items], fontsize=7.5)
            if r == 0:
                ax.set_title(title, color=INK, fontsize=9.5, loc="left")
            if c == 0:
                ax.set_ylabel(f"{base}\nsuccess, right instance", color=INK, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Look-alike stress: success on the right instance (mean and 95 % interval over seeds)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
