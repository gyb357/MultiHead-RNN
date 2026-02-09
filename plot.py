import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt
from train.utils import ClassificationMetrics
from pathlib import Path


# Models
_MODELS = ['RNN', 'LSTM', 'GRU', 'LTC', 'CfC']

# Window
_WINDOW = 5

# Number of columns in the plot grid
_NUM_COLS = 4

# Boxplot Colors
_HEX_START = '#FCB9AA'
_HEX_END = "#35ABA5"


# Main function
def main() -> None:
    # Setup
    columns = list(ClassificationMetrics._fields) + ['run', 'window', 'train_time']
    metrics = columns[:16]

    # Load csv
    files = list(Path('result/').glob('*.csv'))
    print(f"Found {len(files)} csv files.")
    if not files:
        raise FileNotFoundError("No CSV files found in the 'result' directory.")
    
    df_list = []
    for file in files:
        # Extract model name from filename
        filename = file.stem
        model_found = None
        
        # Check if any model name is in the filename
        for model in _MODELS:
            if model.lower() in filename.lower():
                model_found = model
                break
        
        if model_found is None:
            print(f"Warning: No valid model found in filename '{filename}'. Skipping this file.")
            continue
            
        df = pd.read_csv(file, names=columns, header=0)
        df['model'] = model_found
        df_list.append(df)

    # Concatenate all dataframes
    df = pd.concat(df_list, ignore_index=True)

    # Filter by window
    df = df[df['window'] == _WINDOW]

    # Convert metrics to numeric
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

    # Median summary
    print(f"\n[Window={_WINDOW}] Best model(s) by median for each metric\n" + "-"*70)
    for metric in metrics:
        medians = (
            df
            .groupby('model')[metric]
            .median()
            .dropna()
        )
        if medians.empty:
            print(f"{metric.upper():<20} | No data available")
            continue

        best_score = medians.max()
        best_models = medians[medians == best_score].index.tolist()
        best_models_str = ", ".join(best_models)

        print(f"{metric.upper():<20} | best median = {best_score:.3f} | model(s): {best_models_str}")
    print("-"*70 + "\n")

    # Generate pastel colors for each model
    start_rgb = np.array(mcolors.to_rgb(_HEX_START))
    end_rgb   = np.array(mcolors.to_rgb(_HEX_END))

    n_models = len(_MODELS)
    if n_models > 1:
        pastel_colors = [
            mcolors.to_hex(start_rgb + (end_rgb - start_rgb) * i/(n_models-1))
            for i in range(n_models)
        ]
    else:
        pastel_colors = [_HEX_START]

    # Draw boxplots
    nrows = math.ceil(len(metrics) / _NUM_COLS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=_NUM_COLS,
        figsize=(6*_NUM_COLS, 6*nrows),
        constrained_layout=False
    )
    fig.suptitle(f'Model Comparison on Window={_WINDOW} Across Metrics', fontsize=14)
    axes = axes.flatten()

    # Create boxplots for each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = [df[df['model'] == m][metric].dropna() for m in _MODELS]
        bp = ax.boxplot(data, patch_artist=True)
        for patch, color in zip(bp['boxes'], pastel_colors):
            patch.set_facecolor(color)
        ax.set_title(metric.upper(), fontsize=10)
        ax.set_xticklabels(_MODELS, fontsize=8, rotation=45, ha='right')
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

