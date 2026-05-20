import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, prediction_domain, grid_x, grid_y, title, metrics=None):
    """
    Plots a decision boundary with high-quality styling and a perfectly centered metrics subtitle.
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
        'grid.linestyle': '--'
    }
    
    with plt.rc_context(custom_rc):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # True Positive Data points
        ax.scatter(
            X[y == 1]['x'], X[y == 1]['y'],
            color='blue', marker='o', edgecolors='k', label='Positive', zorder=3
        )

        # True Negative Data points
        ax.scatter(
            X[y == 0]['x'], X[y == 0]['y'],
            color='red', marker='o', edgecolors='k', label='Negative', zorder=3
        )

        # Decision boundary
        Z = model.predict(prediction_domain).reshape(grid_x.shape)
        ax.contourf(grid_x, grid_y, Z, alpha=0.2, cmap=plt.cm.RdBu, zorder=0)
        
        # Predicted points overlay
        ax.scatter(
            X['x'], X['y'], 
            edgecolors='y', marker='.', color='yellow', s=10, 
            label='Points Predicted', zorder=4
        )

        # Labels & Grid
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', framealpha=0.8)
        ax.grid(True)

        # --- Perfect Centering Logic for Titles ---
        if metrics:
            # Format the metrics string
            metrics_str = " | ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            
            # Place the main title using ax.set_title (anchored to the center of the axes)
            ax.set_title(title, fontsize=13, weight='bold', pad=22)
            
            # Place the subtitle slightly lower, perfectly centered on the axes (x=0.5)
            # transform=ax.transAxes ensures coordinates (0.5, 1.0) mean top-center of the plot
            ax.text(0.5, 1.02, metrics_str, transform=ax.transAxes, 
                    fontsize=10, color='dimgray', ha='center', va='bottom')
        else:
            ax.set_title(title, pad=12, weight='bold')

        # Adjust layout securely without messing up coordinates
        fig.tight_layout()
        
        plt.close(fig)

    return fig, ax
