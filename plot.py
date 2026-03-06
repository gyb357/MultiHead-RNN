import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt
from train.utils import ClassificationMetrics
from pathlib import Path


# Window
_WINDOW = 5

# Number of columns in the plot grid
_NUM_COLS = 4

# Boxplot Colors
_HEX_START = '#FCB9AA'
_HEX_END = "#35ABA5"


def main() -> None:
    # Setup
    columns = list(ClassificationMetrics._fields) + ['run', 'window', 'train_time']
    metrics = columns[:16]

    # Load CSV files
    files = list(Path('plot/').glob('*.csv'))
    print(f"Found {len(files)} CSV files.")
    if not files:
        raise FileNotFoundError("No CSV files found in the 'plot' directory.")

    df_list = []
    valid_files = []

    for file in files:
        try:
            df_temp = pd.read_csv(file, names=columns, header=0)
        except Exception as e:
            print(f"Warning: Could not read '{file.name}': {e}. Skipping.")
            continue

        # Validate that all expected columns are present
        missing_cols = [col for col in columns if col not in df_temp.columns]
        if missing_cols:
            print(f"Warning: '{file.name}' is missing columns {missing_cols}. Skipping.")
            continue

        label = file.stem  # Use filename (without extension) as identifier
        df_temp['label'] = label
        df_list.append(df_temp)
        valid_files.append(label)
        print(f"  Loaded: '{file.name}' → label='{label}'")

    if not df_list:
        raise ValueError("No valid CSV files could be loaded.")

    # All unique labels (preserving load order)
    labels = list(dict.fromkeys(valid_files))

    # Concatenate all dataframes
    df = pd.concat(df_list, ignore_index=True)
    # Filter by window
    df = df[df['window'] == _WINDOW]
    # Convert metrics to numeric
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

    # Median summary
    print(f"\n[Window={_WINDOW}] Best file(s) by median for each metric\n" + "-"*70)
    for metric in metrics:
        medians = (
            df
            .groupby('label')[metric]
            .median()
            .dropna()
        )
        if medians.empty:
            print(f"{metric.upper():<20} | No data available")
            continue
        
        best_score = medians.max()
        best_labels = medians[medians == best_score].index.tolist()
        best_labels_str = ", ".join(best_labels)
        print(f"{metric.upper():<20} | best median = {best_score:.3f} | file(s): {best_labels_str}")
    print("-"*70 + "\n")

    # Generate colors for each label
    start_rgb = np.array(mcolors.to_rgb(_HEX_START))
    end_rgb   = np.array(mcolors.to_rgb(_HEX_END))
    n_labels = len(labels)
    if n_labels > 1:
        colors = [
            mcolors.to_hex(start_rgb + (end_rgb - start_rgb) * i / (n_labels - 1))
            for i in range(n_labels)
        ]
    else:
        colors = [_HEX_START]

    # Draw boxplots
    nrows = math.ceil(len(metrics) / _NUM_COLS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=_NUM_COLS,
        figsize=(6 * _NUM_COLS, 6 * nrows),
        constrained_layout=False
    )
    fig.suptitle(f'File Comparison on Window={_WINDOW} Across Metrics', fontsize=14)
    axes = axes.flatten()

    # Create boxplots for each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = [df[df['label'] == lbl][metric].dropna() for lbl in labels]
        bp = ax.boxplot(data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(metric.upper(), fontsize=10)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', linewidth=0.4)

    # Hide unused axes
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    # Show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()

