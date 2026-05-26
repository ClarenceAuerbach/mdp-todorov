# plotting.py
"""
All visualisation functions for the Todorov LMDP experiment.
Keeps main.py clean: computation only, no plt calls there.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def _values_to_grid(values, state_idx, rows, cols):
    """Map a flat value array to a (rows × cols) image (NaN for walls)."""
    grid = np.full((rows, cols), np.nan)
    for s, i in state_idx.items():
        grid[s] = values[i]
    return grid


def plot_value_maze(ax, values, state_idx, rows, cols, absorbing,
                    title, cmap='viridis_r', colorbar_label='Cost-to-go'):
    """
    Draw a heat-map of a value function on the maze grid.

    Parameters
    ----------
    ax          : matplotlib Axes
    values      : ndarray (n_states,)
    state_idx   : dict  (row, col) → index
    rows, cols  : int  grid dimensions
    absorbing   : list of (row, col) absorbing-state coordinates
    title       : str  axes title
    cmap        : colormap name
    colorbar_label : str
    """
    grid = _values_to_grid(values, state_idx, rows, cols)
    ax.set_facecolor('#222222')
    im = ax.imshow(grid, cmap=cmap, origin='upper', interpolation='none')
    ax.set_title(title, fontweight='bold')
    for r_abs, c_abs in absorbing:
        ax.text(c_abs, r_abs, '★', color='red',
                ha='center', va='center', fontsize=18, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.85)
    return im



def plot_convergence(ax, sample_steps, err_z, err_q):
    """MSE error trace for Z-learning vs Q-learning."""
    ax.plot(sample_steps, err_z, label='Z-Learning', marker='o', markersize=4)
    ax.plot(sample_steps, err_q, label='Q-Learning', marker='s', markersize=4)
    ax.set_title('Solution model-based', fontweight='bold')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Normalised max error')
    ax.legend()
    ax.grid(True, alpha=0.3)



def show_all_results(v_true, v_z, v_q,
                     sample_steps, err_z, err_q,
                     state_idx, rows, cols, absorbing):
    """
    2 × 2 figure:
      [Ground truth (z-iteration)]  [Convergence error]
      [Z-Learning value map      ]  [Q-Learning value map]

    Parameters
    ----------
    v_true       : ndarray  ground-truth cost-to-go from model-based z-iteration
    v_z, v_q     : ndarray  final learned value functions
    sample_steps : array    x-axis for error plot
    err_z, err_q : lists    MSE at each sample step
    state_idx    : dict  (row,col) → int
    rows, cols   : int
    absorbing    : list of (row, col)
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    ax_z    = fig.add_subplot(gs[1, 0])
    ax_q    = fig.add_subplot(gs[1, 1])

    plot_value_maze(ax_gt, v_true, state_idx, rows, cols, absorbing,
                    title='Value iteration  (model-based)',
                    cmap='plasma', colorbar_label='Cost-to-go')

    plot_convergence(ax_err, sample_steps, err_z, err_q)

    # Normalise learned values for visual consistency with ground truth
    def norm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return (v - lo) / (hi - lo + 1e-10)

    plot_value_maze(ax_z, norm(v_z), state_idx, rows, cols, absorbing,
                    title='Z-Learning  (model-free)',
                    cmap='plasma', colorbar_label='Normalised cost')

    plot_value_maze(ax_q, norm(v_q), state_idx, rows, cols, absorbing,
                    title='Q-Learning  (model-free)',
                    cmap='plasma', colorbar_label='Normalised cost')

    fig.suptitle('Linearly-Solvable MDPs — Todorov (2006)',
                 fontsize=14, fontweight='bold')
    plt.show()
    return fig
