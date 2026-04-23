import matplotlib.pyplot as plt
import pandas as pd


def plot_in_rectangular_coordinates(X: pd.DataFrame, y: pd.DataFrame, title: str, axis_x='x', axis_y='y', figsize=(8, 6)):
    pos = X[y==1]
    neg = X[y==0]

    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)

    ax.scatter(pos[axis_x], pos[axis_y], color='blue', marker='o', label='Positive')
    ax.scatter(neg[axis_x], neg[axis_y], color='red', marker='o', label='Negative')

    ax.set_title(title)
    ax.set_xlabel(axis_x)
    ax.set_ylabel(axis_y)
    ax.legend()

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
