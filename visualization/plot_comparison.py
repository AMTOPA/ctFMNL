"""Runtime and paired-speedup figures for Section 6.2."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "comparison"
PAPER_FIGURES = ROOT / "figures" / "comparison"
METHODS = ("FABS", "Guan_A1_local")
HARD_LIMIT_SECONDS = 600.0
N_INSTANCES = 20


def save_figure(fig, stem):
    """Save PDF and PNG figures."""
    for suffix, dpi in (('pdf', None), ('png', 400)):
        kwargs = {'bbox_inches': 'tight'}
        if dpi:
            kwargs['dpi'] = dpi
        output = PAPER_FIGURES / f'{stem}.{suffix}'
        fig.savefig(output, **kwargs)


def export_figures(summary, paired_summary):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    })
    colors = {'FABS': '#0072B2', 'Guan_A1_local': '#D55E00'}
    names = {'FABS': 'FABS', 'Guan_A1_local': 'Guan A.1 (local)'}
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    # Use equally spaced positions for the observed scales.  This avoids the
    # unreadable cluster of tick labels caused by a log x-axis spanning 100--100,000,
    # while preserving the exact scale labels in the data table and caption.
    runtime_scales = sorted(summary.loc[summary.completed > 0, 'n'].unique())
    runtime_pos = {int(n): i for i, n in enumerate(runtime_scales)}
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for method in METHODS:
        group = summary[(summary.algorithm == method) & (summary.completed > 0)].sort_values('n')
        if group.empty:
            continue
        x = group.n.map(runtime_pos).to_numpy(float)
        median = group.median_s.to_numpy(float)
        low = median - group.q1_s.to_numpy(float)
        high = group.q3_s.to_numpy(float) - median
        # Draw the IQR whiskers underneath a compact marker so the quartile
        # bars remain visible, especially at the small scales.
        ax.errorbar(x, median, yerr=np.vstack([low, high]), fmt='none',
                    ecolor=colors[method], elinewidth=0.95, capsize=2.2,
                    capthick=0.95, alpha=0.9, zorder=2)
        ax.plot(x, median, color=colors[method], marker='o', markersize=3.1,
                markeredgewidth=0.55, linewidth=1.45, label=names[method],
                zorder=3)

    # A hard timeout is plotted only when the other method has a completed call
    # at the same scale.  Both-method cutoffs are intentionally omitted rather
    # than being presented as measured runtimes.
    hard_timeout_rows = summary[summary.hard_timeouts > 0]
    plotted_timeout = None
    for row in hard_timeout_rows.itertuples(index=False):
        n = int(row.n)
        if n not in runtime_pos:
            continue
        other = summary[(summary.n == n) & (summary.algorithm != row.algorithm)]
        if other.empty or int(other.iloc[0].completed) == 0:
            continue
        x = runtime_pos[n] + (-0.08 if row.algorithm == 'FABS' else 0.08)
        ax.scatter([x], [HARD_LIMIT_SECONDS], marker='v', s=42,
                   facecolors='white', edgecolors=colors[row.algorithm],
                   linewidths=1.2, zorder=4)
        ax.annotate(f'{int(row.hard_timeouts)}/{N_INSTANCES}', (x, HARD_LIMIT_SECONDS),
                    xytext=(0, -13), textcoords='offset points', ha='center', va='top',
                    fontsize=7, color=colors[row.algorithm])
        if plotted_timeout is None:
            plotted_timeout = row

    ax.set_xticks(range(len(runtime_scales)), labels=[f'{int(n):,}' for n in runtime_scales])
    ax.set_xlim(-0.35, max(len(runtime_scales) - 1, 0) + 0.35)
    ax.set_xlabel('Number of products, $n$')
    ax.set_ylabel('Solver runtime (seconds)')
    ax.set_yscale('log')
    ax.set_ylim(bottom=max(1e-5, float(summary.loc[summary.completed > 0, 'q1_s'].min()) * 0.65),
                top=HARD_LIMIT_SECONDS * 1.45)
    ax.grid(True, which='both', linestyle=':', linewidth=.65, alpha=.48)
    handles, labels = ax.get_legend_handles_labels()
    if plotted_timeout is not None:
        timeout = plotted_timeout
        timeout_handle = Line2D([0], [0], marker='v', linestyle='None', markersize=5,
                                markerfacecolor='white', markeredgecolor=colors[timeout.algorithm],
                                label='Hard timeout (600 s)')
        handles.append(timeout_handle)
        labels.append(timeout_handle.get_label())
    ax.legend(handles, labels, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_figure(fig, 'figure_6_2_runtime')
    plt.close(fig)

    speed_group = paired_summary[paired_summary.paired_instances > 0].sort_values('n')
    speed_scales = [int(n) for n in speed_group.n]
    speed_pos = {n: i for i, n in enumerate(speed_scales)}
    fig, ax = plt.subplots(figsize=(7.0, 3.7), constrained_layout=True)
    if not speed_group.empty:
        x = speed_group.n.map(speed_pos).to_numpy(float)
        median = speed_group.median_speedup.to_numpy(float)
        low = median - speed_group.q1_speedup.to_numpy(float)
        high = speed_group.q3_speedup.to_numpy(float) - median
        # Keep the square marker compact so the IQR caps remain visible.
        ax.errorbar(x, median, yerr=np.vstack([low, high]), fmt='none',
                    ecolor='#009E73', elinewidth=0.95, capsize=2.2,
                    capthick=0.95, alpha=0.9, zorder=2)
        ax.plot(x, median, color='#009E73', marker='s', markersize=3.0,
                markeredgewidth=0.55, linewidth=1.45,
                label='Paired median and IQR', zorder=3)
    ax.axhline(1.0, color='#555555', linewidth=1, linestyle='--', label='Equal runtime')
    ax.set_xticks(range(len(speed_scales)), labels=[f'{n:,}' for n in speed_scales])
    ax.set_xlim(-0.35, max(len(speed_scales) - 1, 0) + 0.35)
    ax.set_xlabel('Number of products, $n$')
    ax.set_ylabel('Paired speedup (Guan A.1 / FABS)')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle=':', linewidth=.65, alpha=.48)
    ax.legend(frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_figure(fig, 'figure_6_2_speedup')
    plt.close(fig)


def main():
    summary = pd.read_csv(DATA / "runtime_summary.csv")
    paired = pd.read_csv(DATA / "speedup_summary.csv")
    export_figures(summary, paired)
    print("Wrote figures to", PAPER_FIGURES)


if __name__ == "__main__":
    main()
