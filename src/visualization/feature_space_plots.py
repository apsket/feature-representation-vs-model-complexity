import matplotlib.pyplot as plt
import pandas as pd


def plot_in_rectangular_coordinates(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    title: str, 
    axis_x='x1', 
    axis_y='x2', 
    axis_x_label=None,  # Custom display label for X axis
    axis_y_label=None,  # Custom display label for Y axis
    figsize=(8, 6), 
    metrics=None
):
    """
    Plots data points in rectangular coordinates.
    Separates the DataFrame column keys from the visual display labels to prevent KeyErrors.
    """
    custom_rc = {
        'font.family': 'serif',
        'text.usetex': False,          
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'grid.alpha': 0.20,
        'grid.linestyle': '--',
        'mathtext.fontset': 'dejavuserif' 
    }
    
    # Use column names to extract the data safely
    pos = X[y == 1]
    neg = X[y == 0]

    with plt.rc_context(custom_rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        # Reference data columns safely using axis_x and axis_y internal names
        ax.scatter(pos[axis_x], pos[axis_y], color='blue', marker='o', edgecolors='k', label='Positive', zorder=3)
        ax.scatter(neg[axis_x], neg[axis_y], color='red', marker='o', edgecolors='k', label='Negative', zorder=3)

        # Set display labels (fallback to column names if no custom label is provided)
        ax.set_xlabel(axis_x_label if axis_x_label is not None else axis_x)
        ax.set_ylabel(axis_y_label if axis_y_label is not None else axis_y)
        
        ax.legend(loc='best', framealpha=0.8)
        ax.grid(True)

        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            ax.set_title(title, fontsize=13, weight='bold', pad=22)
            ax.text(0.5, 1.02, metrics_str, transform=ax.transAxes, fontsize=10, color='dimgray', ha='center', va='bottom')
        else:
            ax.set_title(title, pad=12, weight='bold')

        fig.tight_layout()
        plt.close(fig)

    return fig, ax


class PlotManager:
    def __init__(self, config):
        self.config = config

    def handle(self, fig, name):
        if self.config["save"]:
            fig.savefig(f"{self.config['out_dir']}/{name}.png",
                        dpi=300, bbox_inches="tight")

        if self.config["show"]:
            plt.show()
        else:
            plt.close(fig)
