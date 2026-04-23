import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, prediction_domain, grid_x, grid_y, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points (true labels)
    ax.scatter(
        X[y == 1]['x'], X[y == 1]['y'],
        color='blue', marker='o', edgecolors='k', label='Positive', zorder=2
    )

    ax.scatter(
        X[y == 0]['x'], X[y == 0]['y'],
        color='red', marker='o', label='Negative', zorder=2
    )

    # decision boundary
    Z = model.predict(prediction_domain).reshape(grid_x.shape)
    ax.contourf(grid_x, grid_y, Z, alpha=0.2, cmap=plt.cm.RdBu, zorder=0)
    ax.scatter(X['x'], X['y'], edgecolors='y', marker='.', color='yellow', s=10, label='Points Predicted', zorder=4)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

    plt.close(fig)

    return fig, ax
