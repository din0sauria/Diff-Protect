# Loss Curve Plotting Script for Diff-Protect
#
# Usage:
#   # Plot all modes for a specific image under out_sd3/
#   python code/plot_loss.py --root out_sd3 --image suzume
#
#   # Plot specific modes
#   python code/plot_loss.py --root out_sd3 --image suzume --modes O A B C D
#
#   # Plot with custom output path and DPI
#   python code/plot_loss.py --root out_sd3 --image suzume --output figures/ --dpi 300
#
#   # Plot all images found under out_sd3/
#   python code/plot_loss.py --root out_sd3 --all
#
#   # Compare across different g_mode / epsilon settings
#   python code/plot_loss.py --root out_sd3 --image suzume --group_by mode

import argparse
import os
import glob
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm


def _configure_chinese_font():
    """Auto-detect and configure a CJK-capable font for matplotlib."""
    # Priority list of known CJK fonts
    cjk_font_names = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'PingFang SC',
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
        'Source Han Sans SC', 'AR PL UMing CN', 'Droid Sans Fallback',
    ]
    for name in cjk_font_names:
        try:
            prop = fm.FontProperties(family=name)
            if prop.get_name() != 'DejaVu Sans':
                plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return name
        except Exception:
            continue

    # Fallback: search for CJK .ttf/.ttc files on disk
    for path in fm.findSystemFonts():
        try:
            f = fm.FontProperties(fname=path)
            name = f.get_name()
            if any(k in name.lower() for k in ['cjk', 'droid', 'simhei', 'noto sans sc', 'wqy']):
                plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return name
        except Exception:
            continue
    return None

# ---- Style ----
MODE_COLORS = {
    'O': '#7f7f7f',   # gray (baseline)
    'A': '#e63946',   # red
    'B': '#457b9d',   # blue
    'C': '#2a9d8f',   # teal
    'D': '#e9c46a',   # yellow
}

MODE_LABELS = {
    'O': 'O: Textual Loss Only (baseline)',
    'A': 'A: Cross-Modal Alignment Disruption',
    'B': 'B: Attention Feature Shift',
    'C': 'C: Temporal Consistency Break',
    'D': 'D: Modality Imbalance',
}

MODE_LABELS_ZH = {
    'O': 'O: 仅纹理损失（基线）',
    'A': 'A: 跨模态对齐破坏',
    'B': 'B: 注意力特征偏移',
    'C': 'C: 时序一致性破坏',
    'D': 'D: 模态不平衡',
}

LINESTYLES = {
    'O': '--',
    'A': '-',
    'B': '-',
    'C': '-',
    'D': '-',
}


def parse_dirname(dirname):
    """Parse experiment config from directory name.

    e.g. 'A_eps16_steps100_gmode+_tw1.0_mw1.0' ->
         {'mode': 'A', 'epsilon': 16, 'steps': 100, 'g_mode': '+', 'textual_weight': 1.0, 'mmdit_weight': 1.0}
    """
    config = {}
    parts = dirname.split('_')
    # mode is always first
    config['mode'] = parts[0]

    # Parse key-value pairs
    patterns = {
        'epsilon': (r'eps(\d+)', int),
        'steps': (r'steps(\d+)', int),
        'g_mode': (r'gmode([+-])', str),
        'textual_weight': (r'tw([\d.]+)', float),
        'mmdit_weight': (r'mw([\d.]+)', float),
    }
    for key, (pat, dtype) in patterns.items():
        m = re.search(pat, dirname)
        if m:
            val = m.group(1)
            config[key] = dtype(val)
    return config


def find_loss_files(root, image_name=None):
    """Find all _loss.npy files under root directory.

    Returns:
        list of (config_dict, loss_path) tuples
    """
    results = []
    for exp_dir in sorted(os.listdir(root)):
        exp_path = os.path.join(root, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        config = parse_dirname(exp_dir)

        # Find loss files
        if image_name:
            pattern = os.path.join(exp_path, '**', f'{image_name}_loss.npy')
        else:
            pattern = os.path.join(exp_path, '**', '*_loss.npy')

        for loss_path in sorted(glob.glob(pattern, recursive=True)):
            # Extract image name from path
            basename = os.path.basename(loss_path)
            img_name = basename.replace('_loss.npy', '')
            results.append((config, img_name, loss_path))

    return results


def plot_loss_curves(entries, modes=None, output_dir='.', zh=False,
                     figsize=(8, 5), dpi=150, smooth_window=None,
                     log_scale=False, normalize=False, separate=False):
    """Plot loss curves for the given entries.

    Args:
        entries: list of (config, img_name, loss_path) tuples
        modes: list of modes to plot (None = all)
        output_dir: directory to save figures
        zh: use Chinese labels
        figsize: figure size
        dpi: output DPI
        smooth_window: moving average window size (None = no smoothing)
        log_scale: use log scale for y-axis
        normalize: normalize loss to [0, 1] range per curve
        separate: plot each image in a separate figure
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = MODE_LABELS_ZH if zh else MODE_LABELS

    # Configure Chinese font if needed
    if zh:
        font_name = _configure_chinese_font()
        if font_name:
            print(f'Using CJK font: {font_name}')
        else:
            print('Warning: No CJK font found, Chinese text may not render. Falling back to English labels.')
            labels = MODE_LABELS

    # Group by image name
    images = {}
    for config, img_name, loss_path in entries:
        mode = config['mode']
        if modes and mode not in modes:
            continue
        if img_name not in images:
            images[img_name] = []
        images[img_name].append((config, loss_path))

    for img_name, curves in images.items():
        # Sort by mode order
        mode_order = ['O', 'A', 'B', 'C', 'D']
        curves.sort(key=lambda x: mode_order.index(x[0]['mode']) if x[0]['mode'] in mode_order else 99)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for config, loss_path in curves:
            mode = config['mode']
            loss = np.load(loss_path)

            if smooth_window and smooth_window > 1:
                loss = np.convolve(loss, np.ones(smooth_window) / smooth_window, mode='valid')

            if normalize:
                lmin, lmax = loss.min(), loss.max()
                if lmax > lmin:
                    loss = (loss - lmin) / (lmax - lmin)
                else:
                    loss = np.zeros_like(loss)

            steps = np.arange(len(loss))
            color = MODE_COLORS.get(mode, '#333333')
            ls = LINESTYLES.get(mode, '-')
            label = labels.get(mode, f'Mode {mode}')

            # Add config details to label
            eps = config.get('epsilon', '?')
            g = config.get('g_mode', '?')
            label += f' (ε={eps}, g={g})'

            ax.plot(steps, loss, color=color, linestyle=ls, linewidth=1.8, label=label, alpha=0.9)

        xlabel = '迭代步数 (PGD Step)' if zh else 'PGD Step'
        ylabel = '归一化损失' if (zh and normalize) else ('Normalized Loss' if normalize else 'Loss')
        title = f'{img_name} — 对抗攻击损失曲线' if zh else f'{img_name} — Adversarial Attack Loss Curves'

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{img_name}_loss_curve.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {save_path}')
        plt.close(fig)

    # ---- Summary comparison plot (all images aggregated) ----
    if len(images) > 1 and not separate:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for mode in mode_order:
            all_losses = []
            for img_name, curves in images.items():
                for config, loss_path in curves:
                    if config['mode'] == mode:
                        loss = np.load(loss_path)
                        if smooth_window and smooth_window > 1:
                            loss = np.convolve(loss, np.ones(smooth_window) / smooth_window, mode='valid')
                        if normalize:
                            lmin, lmax = loss.min(), loss.max()
                            if lmax > lmin:
                                loss = (loss - lmin) / (lmax - lmin)
                            else:
                                loss = np.zeros_like(loss)
                        all_losses.append(loss)

            if not all_losses:
                continue

            # Average across images
            min_len = min(len(l) for l in all_losses)
            avg_loss = np.mean([l[:min_len] for l in all_losses], axis=0)
            std_loss = np.std([l[:min_len] for l in all_losses], axis=0)
            steps = np.arange(min_len)

            color = MODE_COLORS.get(mode, '#333333')
            ls = LINESTYLES.get(mode, '-')
            label = labels.get(mode, f'Mode {mode}')

            ax.plot(steps, avg_loss, color=color, linestyle=ls, linewidth=2.0, label=label)
            ax.fill_between(steps, avg_loss - std_loss, avg_loss + std_loss,
                            color=color, alpha=0.15)

        xlabel = '迭代步数 (PGD Step)' if zh else 'PGD Step'
        ylabel = '归一化损失' if (zh and normalize) else ('Normalized Loss' if normalize else 'Loss')
        title = '汇总对比 — 对抗攻击损失曲线' if zh else 'Summary — Adversarial Attack Loss Curves'

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'summary_loss_curve.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {save_path}')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot loss curves for Diff-Protect adversarial attacks')
    parser.add_argument('--root', type=str, default='out_sd3',
                        help='Root directory containing experiment outputs (default: out_sd3)')
    parser.add_argument('--image', type=str, default=None,
                        help='Specific image name to plot (without _loss.npy suffix)')
    parser.add_argument('--all', action='store_true',
                        help='Plot all images found under root')
    parser.add_argument('--modes', nargs='+', default=None,
                        choices=['O', 'A', 'B', 'C', 'D'],
                        help='Modes to plot (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for figures (default: <root>/figures/)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure DPI (default: 150)')
    parser.add_argument('--smooth', type=int, default=None,
                        help='Moving average window size for smoothing')
    parser.add_argument('--log', action='store_true',
                        help='Use log scale for y-axis')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize each curve to [0, 1]')
    parser.add_argument('--zh', action='store_true',
                        help='Use Chinese labels')
    parser.add_argument('--separate', action='store_true',
                        help='Do not create summary plot for multiple images')

    args = parser.parse_args()

    if not args.all and args.image is None:
        parser.error('Specify --image <name> or --all')

    entries = find_loss_files(args.root, image_name=args.image)
    if not entries:
        print(f'No loss files found under {args.root}')
        return

    print(f'Found {len(entries)} loss file(s):')
    for config, img_name, loss_path in entries:
        loss = np.load(loss_path)
        print(f'  [{config["mode"]}] {img_name}: {len(loss)} steps, '
              f'range=[{loss.min():.4f}, {loss.max():.4f}]')

    output_dir = args.output or os.path.join(args.root, 'figures')

    plot_loss_curves(
        entries,
        modes=args.modes,
        output_dir=output_dir,
        zh=args.zh,
        dpi=args.dpi,
        smooth_window=args.smooth,
        log_scale=args.log,
        normalize=args.normalize,
        separate=args.separate,
    )


if __name__ == '__main__':
    main()
