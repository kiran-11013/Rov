"""Static result figures (matplotlib). Colours: validated categorical palette, fixed slot order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e2db", "#fcfbf8"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def persistence_curves(ctx, path, classes=("mug", "chair", "laptop", "bag", "toolbox", "box", "monitor")):
    """Empirical P(still on mapped place) vs time since seen (test weeks), with our fitted prior."""
    from .priors import labelled_pairs
    grid = np.array([0.25, 0.5, 1, 2, 4, 8, 24, 48, 72, 168])
    rows = labelled_pairs(ctx.history, ctx.world, ctx.train_end, ctx.test_end, tuple(grid), step_h=2.0)
    fig, ax = _fig()
    for i, c in enumerate(classes):
        col = SERIES[i % len(SERIES)]
        emp, fit = [], []
        for g in grid:
            sel = [r for r in rows if r[1] == c and r[3] == g]
            emp.append(np.mean([r[4] for r in sel]) if sel else np.nan)
            fit.append(np.mean(ctx.ours.predict_many(sel)) if sel else np.nan)
        ax.plot(grid, fit, color=col, linewidth=2, label=c)
        ax.plot(grid, emp, "o", color=col, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 4, 24, 72, 168])
    ax.set_xticklabels(["15 min", "1 h", "4 h", "1 day", "3 days", "1 week"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time since the object was mapped", color=INK)
    ax.set_ylabel("P(still within 1 m of mapped place)", color=INK)
    ax.set_title("Object memory expires at very different rates by class\n"
                 "lines: fitted prior (ours) · dots: held-out ground truth", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decision_regions(dr, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title in ((axes[0], "full", "Altitudes 0.4 / 1.0 / 1.8 m"), (axes[1], "fixed", "Fixed 0.4 m (ground-robot equivalent)")):
        _style(ax)
        d = dr[key]
        for i, a in enumerate(("trust", "verify", "search")):
            ax.plot(d["p"], d["costs"][a], color=SERIES[i], linewidth=2, label=a)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("P(object still on its mapped place)", color=INK)
    axes[0].set_ylabel("Expected cost of first action (s)", color=INK)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Which first action is cheapest, as memory confidence changes (laptop on a far desk, 1 day)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def success_vs_budget(curves, budgets, path, arms):
    fig, ax = _fig()
    for i, name in enumerate(arms):
        ax.plot(budgets, curves[name], color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4,
                label=name)
    ax.set_xlabel("Time budget per command (s)", color=INK)
    ax.set_ylabel("Success (right instance, within budget)", color=INK)
    ax.set_title("Success within a time budget, all commands", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def risk_coverage(e3, path):
    fig, ax = _fig(6.5, 4.0)
    for i, (name, r) in enumerate(e3.items()):
        cov, risk = r["risk_coverage"]
        ax.plot(cov, risk, color=SERIES[i], linewidth=2, label=name)
    ax.set_xlabel("Coverage (fraction of commands acted on, most confident first)", color=INK)
    ax.set_ylabel("Error rate among acted commands", color=INK)
    ax.set_title("Grounding under stale memory: risk vs coverage", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def floor_plan(world, drone, path):
    """Layout with occluders, places and viewpoints coloured by how many places they see at 1.8 m."""
    fig, ax = _fig(9, 4.4)
    ax.grid(False)
    for (a, b) in world.walls:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, linewidth=3)
    ax.plot([0, 18, 18, 0, 0], [0, 0, 8, 8, 0], color=INK, linewidth=1.5)
    for bx in world.boxes:
        col = "#c3c2b7" if bx.h < 1.0 else ("#86b6ef" if bx.h < 1.9 else "#5f5e58")
        ax.add_patch(matplotlib.patches.Rectangle((bx.x0, bx.y0), bx.x1 - bx.x0, bx.y1 - bx.y0, color=col))
    kinds = {"floor": ("o", SERIES[1]), "table": ("s", SERIES[2]), "shelf": ("^", SERIES[6])}
    for k, (mk, col) in kinds.items():
        pts = np.array([(p.x, p.y) for p in world.places if p.kind == k])
        ax.plot(pts[:, 0], pts[:, 1], mk, color=col, markersize=4, label=f"{k} place", linestyle="none")
    ax.plot(*world.dock, "*", color=SERIES[7], markersize=12, label="dock", linestyle="none")
    for name, (x0, _y0, x1, _y1) in world.rooms.items():
        ax.text((x0 + x1) / 2, 8.25, name, ha="center", color=MUTED, fontsize=9)
    ax.set_xlim(-0.2, 18.2)
    ax.set_ylim(-0.2, 8.6)
    ax.set_aspect("equal")
    ax.set_title(f"Simulated space ({getattr(world, 'layout', 'open')} layout): desks (light grey), "
                 "1.3-1.4 m partitions (blue), 2 m shelves (dark grey)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
