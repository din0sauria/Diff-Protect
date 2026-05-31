# Loss Curve Plotting Script for Diff-Protect
#
# Supports both legacy .npy (total loss only) and new .npz (component losses).
#
# Usage:
#   # Plot all modes for a specific image
#   python code/plot_loss.py --root out_sd3 --image suzume
#
#   # Plot specific modes
#   python code/plot_loss.py --root out_sd3 --image suzume --modes O A B C D
#
#   # Plot all images
#   python code/plot_loss.py --root out_sd3 --all
#
#   # Normalize + smooth + high DPI
#   python code/plot_loss.py --root out_sd3 --image suzume --normalize --smooth 5 --dpi 300
#
#   # Compare component losses (textual vs mmdit) for one mode
#   python code/plot_loss.py --root out_sd3 --image suzume --modes A --components

import argparse
import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- Style ----
MODE_COLORS = {
    'O': '#7f7f7f',   # gray (baseline)
    'A': '#e63946',   # red
    'B': '#457b9d',   # blue
    'C': '#2a9d8f',   # teal
    'D': '#e9c46a',   # yellow
}

MODE_LABELS = {
    'O': 'O: Textual Only (baseline)',
    'A': 'A: Cross-Modal Disruption',
    'B': 'B: Feature Shift',
    'C': 'C: Temporal Break',
    'D': 'D: Modality Imbalance',
}

LINESTYLES = {
    'O': '--',
    'A': '-',
    'B': '-',
    'C': '-',
    'D': '-',
}

COMPONENT_COLORS = {
    'total': '#333333',
    'textual': '#457b9d',
    'mmdit': '#e63946',
}

COMPONENT_LABELS = {
    'total': 'Total Loss',
    'textual': 'Textual Loss (VAE latent push)',
    'mmdit': 'MMDiT Loss',
}

COMPONENT_LINESTYLES = {
    'total': '-',
    'textual': '--',
    'mmdit': '-.',
}


def parse_dirname(dirname):
    """Parse experiment config from directory name.

    e.g. 'A_eps16_steps100_gmode+_tw1.0_mw1.0' ->
         {'mode': 'A', 'epsilon': 16, 'steps': 100, ...}
    """
    config = {}
    parts = dirname.split('_')
    config['mode'] = parts[0]

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
            config[key] = dtype(m.group(1))
    return config


def load_loss_data(loss_path):
    """Load loss data from .npz (component) or .npy (total only) file.

    Returns dict with keys 'total', 'textual', 'mmdit' (each a 1-D array).
    """
    if loss_path.endswith('.npz'):
        data = np.load(loss_path)
        result = {}
        for key in ['total', 'textual', 'mmdit']:
            result[key] = data[key] if key in data else np.zeros_like(data.get('total', []))
        return result
    else:
        # Legacy .npy: only total loss
        arr = np.load(loss_path)
        return {'total': arr, 'textual': np.zeros_like(arr), 'mmdit': np.zeros_like(arr)}


def find_loss_files(root, image_name=None):
    """Find all loss files under root directory.

    Returns list of (config_dict, img_name, loss_path) tuples.
    """
    results = []
    for exp_dir in sorted(os.listdir(root)):
        exp_path = os.path.join(root, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        config = parse_dirname(exp_dir)

        # Search for both .npz and .npy
        if image_name:
            patterns = [
                os.path.join(exp_path, '**', f'{image_name}_loss.npz'),
                os.path.join(exp_path, '**', f'{image_name}_loss.npy'),
            ]
        else:
            patterns = [
                os.path.join(exp_path, '**', '*_loss.npz'),
                os.path.join(exp_path, '**', '*_loss.npy'),
            ]

        for pattern in patterns:
            for loss_path in sorted(glob.glob(pattern, recursive=True)):
                basename = os.path.basename(loss_path)
                # Remove _loss.npz or _loss.npy
                img_name = re.sub(r'_loss\.(npz|npy)$', '', basename)
                results.append((config, img_name, loss_path))

    # Deduplicate: prefer .npz over .npy for the same image+mode
    seen = {}
    for config, img_name, loss_path in results:
        key = (config['mode'], img_name)
        if key not in seen or loss_path.endswith('.npz'):
            seen[key] = (config, img_name, loss_path)
    return list(seen.values())


def _smooth(arr, window):
    if window and window > 1:
        return np.convolve(arr, np.ones(window) / window, mode='valid')
    return arr


def _normalize(arr):
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


def plot_mode_comparison(images, modes=None, output_dir='.', figsize=(8, 5),
                         dpi=150, smooth_window=None, log_scale=False,
                         normalize=False):
    """Plot total loss curves comparing different modes for each image."""
    for img_name, curves in images.items():
        mode_order = ['O', 'A', 'B', 'C', 'D']
        curves_sorted = sorted(curves, key=lambda x: mode_order.index(x[0]['mode'])
                               if x[0]['mode'] in mode_order else 99)

        fig, ax = plt.subplots(figsize=figsize)
        for config, loss_data in curves_sorted:
            mode = config['mode']
            if modes and mode not in modes:
                continue

            loss = _smooth(loss_data['total'], smooth_window)
            if normalize:
                loss = _normalize(loss)

            steps = np.arange(len(loss))
            ax.plot(steps, loss,
                    color=MODE_COLORS.get(mode, '#333'),
                    linestyle=LINESTYLES.get(mode, '-'),
                    linewidth=1.8, label=MODE_LABELS.get(mode, f'Mode {mode}'),
                    alpha=0.9)

        ax.set_xlabel('PGD Step', fontsize=12)
        ax.set_ylabel('Normalized Loss' if normalize else 'Loss', fontsize=12)
        ax.set_title(f'{img_name} — Adversarial Attack Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax.set_yscale('log')
        plt.tight_layout()

        path = os.path.join(output_dir, f'{img_name}_loss_modes.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {path}')
        plt.close(fig)

    # ---- Summary (average across images) ----
    if len(images) > 1:
        fig, ax = plt.subplots(figsize=figsize)
        for mode in mode_order:
            if modes and mode not in modes:
                continue
            all_losses = []
            for img_name, curves in images.items():
                for config, loss_data in curves:
                    if config['mode'] == mode:
                        loss = _smooth(loss_data['total'], smooth_window)
                        if normalize:
                            loss = _normalize(loss)
                        all_losses.append(loss)
            if not all_losses:
                continue

            min_len = min(len(l) for l in all_losses)
            avg = np.mean([l[:min_len] for l in all_losses], axis=0)
            std = np.std([l[:min_len] for l in all_losses], axis=0)
            steps = np.arange(min_len)

            ax.plot(steps, avg, color=MODE_COLORS.get(mode, '#333'),
                    linestyle=LINESTYLES.get(mode, '-'), linewidth=2.0,
                    label=MODE_LABELS.get(mode, f'Mode {mode}'))
            ax.fill_between(steps, avg - std, avg + std,
                            color=MODE_COLORS.get(mode, '#333'), alpha=0.12)

        ax.set_xlabel('PGD Step', fontsize=12)
        ax.set_ylabel('Normalized Loss' if normalize else 'Loss', fontsize=12)
        ax.set_title('Summary — Adversarial Attack Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax.set_yscale('log')
        plt.tight_layout()

        path = os.path.join(output_dir, 'summary_loss_modes.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {path}')
        plt.close(fig)


def plot_component_comparison(images, modes=None, output_dir='.', figsize=(8, 5),
                              dpi=150, smooth_window=None, log_scale=False,
                              normalize=False):
    """Plot textual / mmdit / total loss components for each mode separately.

    Uses twin y-axes (left=textual, right=mmdit) so that differently-scaled
    components are both readable without normalization.
    """
    mode_order = ['O', 'A', 'B', 'C', 'D']

    for img_name, curves in images.items():
        curves_sorted = sorted(curves, key=lambda x: mode_order.index(x[0]['mode'])
                               if x[0]['mode'] in mode_order else 99)

        n_modes = len([c for c in curves_sorted if (not modes) or c[0]['mode'] in modes])
        if n_modes == 0:
            continue

        fig, axes = plt.subplots(1, n_modes, figsize=(4.5 * n_modes, 5), squeeze=False)
        col = 0

        for config, loss_data in curves_sorted:
            mode = config['mode']
            if modes and mode not in modes:
                continue

            ax = axes[0, col]
            ax2 = ax.twinx()  # right y-axis for mmdit

            # Left axis: total + textual
            for comp_key in ['total', 'textual']:
                arr = _smooth(loss_data[comp_key], smooth_window)
                if normalize:
                    arr = _normalize(arr)
                steps = np.arange(len(arr))
                ax.plot(steps, arr,
                        color=COMPONENT_COLORS[comp_key],
                        linestyle=COMPONENT_LINESTYLES[comp_key],
                        linewidth=1.6,
                        label=COMPONENT_LABELS[comp_key],
                        alpha=0.85)

            # Right axis: mmdit (typically much smaller scale)
            mmdit_arr = _smooth(loss_data['mmdit'], smooth_window)
            if normalize:
                mmdit_arr = _normalize(mmdit_arr)
            steps = np.arange(len(mmdit_arr))
            ax2.plot(steps, mmdit_arr,
                     color=COMPONENT_COLORS['mmdit'],
                     linestyle=COMPONENT_LINESTYLES['mmdit'],
                     linewidth=1.6,
                     label=COMPONENT_LABELS['mmdit'],
                     alpha=0.85)

            ax.set_xlabel('PGD Step', fontsize=10)
            ax.set_ylabel('Loss (total / textual)', fontsize=10, color='#333333')
            ax2.set_ylabel('MMDiT Loss', fontsize=10, color=COMPONENT_COLORS['mmdit'])
            ax.tick_params(axis='y', labelcolor='#333333')
            ax2.tick_params(axis='y', labelcolor=COMPONENT_COLORS['mmdit'])
            ax.set_title(f'Mode {mode}', fontsize=12, fontweight='bold')

            # Merge legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best', framealpha=0.9)

            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if log_scale:
                ax.set_yscale('log')
                ax2.set_yscale('log')

            col += 1

        fig.suptitle(f'{img_name} — Loss Components by Mode', fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = os.path.join(output_dir, f'{img_name}_loss_components.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {path}')
        plt.close(fig)

    # ---- Component comparison across modes (textual vs mmdit side-by-side) ----
    if len(images) > 1:
        for comp_key in ['textual', 'mmdit']:
            fig, ax = plt.subplots(figsize=figsize)
            for mode in mode_order:
                if modes and mode not in modes:
                    continue
                all_losses = []
                for img_name, curves in images.items():
                    for config, loss_data in curves:
                        if config['mode'] == mode:
                            arr = _smooth(loss_data[comp_key], smooth_window)
                            if normalize:
                                arr = _normalize(arr)
                            all_losses.append(arr)
                if not all_losses:
                    continue
                min_len = min(len(l) for l in all_losses)
                avg = np.mean([l[:min_len] for l in all_losses], axis=0)
                std = np.std([l[:min_len] for l in all_losses], axis=0)
                steps = np.arange(min_len)

                ax.plot(steps, avg, color=MODE_COLORS.get(mode, '#333'),
                        linestyle=LINESTYLES.get(mode, '-'), linewidth=1.8,
                        label=MODE_LABELS.get(mode, f'Mode {mode}'))
                ax.fill_between(steps, avg - std, avg + std,
                                color=MODE_COLORS.get(mode, '#333'), alpha=0.12)

            ax.set_xlabel('PGD Step', fontsize=12)
            ax.set_ylabel('Normalized Loss' if normalize else 'Loss', fontsize=12)
            ax.set_title(f'Summary — {COMPONENT_LABELS[comp_key]}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if log_scale:
                ax.set_yscale('log')
            plt.tight_layout()

            path = os.path.join(output_dir, f'summary_{comp_key}_loss.png')
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f'Saved: {path}')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot loss curves for Diff-Protect adversarial attacks')
    parser.add_argument('--root', type=str, default='out_sd3',
                        help='Root directory containing experiment outputs')
    parser.add_argument('--image', type=str, default=None,
                        help='Specific image name (without _loss suffix)')
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
                        help='Moving average window size')
    parser.add_argument('--log', action='store_true',
                        help='Use log scale for y-axis')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize each curve to [0, 1]')
    parser.add_argument('--components', action='store_true',
                        help='Plot textual/mmdit/total component breakdown per mode')

    args = parser.parse_args()

    if not args.all and args.image is None:
        parser.error('Specify --image <name> or --all')

    entries = find_loss_files(args.root, image_name=args.image)
    if not entries:
        print(f'No loss files found under {args.root}')
        return

    # Load loss data
    loaded = []
    for config, img_name, loss_path in entries:
        loss_data = load_loss_data(loss_path)
        has_components = (loss_data['textual'].sum() != 0 or loss_data['mmdit'].sum() != 0)
        loaded.append((config, img_name, loss_path, loss_data, has_components))

        n_steps = len(loss_data['total'])
        comp_info = ''
        if has_components:
            comp_info = f', textual=[{loss_data["textual"].min():.2f}, {loss_data["textual"].max():.2f}], ' \
                        f'mmdit=[{loss_data["mmdit"].min():.4f}, {loss_data["mmdit"].max():.4f}]'
        print(f'  [{config["mode"]}] {img_name}: {n_steps} steps, '
              f'total=[{loss_data["total"].min():.2f}, {loss_data["total"].max():.2f}]'
              f'{comp_info}')

    # Group by image name
    images = {}
    for config, img_name, loss_path, loss_data, _ in loaded:
        if img_name not in images:
            images[img_name] = []
        images[img_name].append((config, loss_data))

    output_dir = args.output or os.path.join(args.root, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Always plot mode comparison (total loss)
    plot_mode_comparison(
        images, modes=args.modes, output_dir=output_dir,
        dpi=args.dpi, smooth_window=args.smooth,
        log_scale=args.log, normalize=args.normalize,
    )

    # Plot component breakdown if requested or if .npz data available
    any_components = any(has_comp for *_, has_comp in loaded)
    if args.components or any_components:
        plot_component_comparison(
            images, modes=args.modes, output_dir=output_dir,
            dpi=args.dpi, smooth_window=args.smooth,
            log_scale=args.log, normalize=args.normalize,
        )

    print(f'\nAll figures saved to: {output_dir}/')


if __name__ == '__main__':
    main()
