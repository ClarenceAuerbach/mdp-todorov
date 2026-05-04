# maze.py
import numpy as np
from scipy.sparse import diags, csr_matrix


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def make_grid_maze(rows, cols, walls=None):
    """Create grid maze with walls."""
    if walls is None:
        walls = set()
    
    states = [(i,j) for i in range(rows) for j in range(cols) if (i,j) not in walls]
    
    def neighbours(i, j):
        candidates = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
        return [(r,c) for r,c in candidates if 0<=r<rows and 0<=c<cols and (r,c) not in walls]
    
    adj = {s: neighbours(*s) for s in states}
    return states, adj




def make_random_maze(rows, cols, wall_prob=0.3, seed=None):
    rng = random.Random(seed)
    all_cells = {(i,j) for i in range(rows) for j in range(cols)}
    
    # randomly block cells, never block (0,0) or (rows-1,cols-1)
    start, goal = (0,0), (rows-1, cols-1)
    walls = {
        (i,j) for (i,j) in all_cells
        if (i,j) not in (start, goal) and rng.random() < wall_prob
    }
    states = list(all_cells - walls)
    
    def grid_neighbours(i, j):
        return [
            (r,c) for r,c in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
            if (r,c) in all_cells - walls
        ]
    
    # BFS connectivity check from start
    visited = {start}
    queue = [start]
    while queue:
        curr = queue.pop(0)
        for nbr in grid_neighbours(*curr):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    
    # if goal unreachable, fall back to no walls
    if goal not in visited:
        walls = set()
        states = list(all_cells)
    
    adj = {c: grid_neighbours(*c) for c in states}
    return states, adj, walls


def show_maze_values(rows, cols, states, sol, idx, walls=None):
    if walls is None:
        walls = set()

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    state_set = set(states)
    
    # colormap scaled to solution range
    finite_vals = [sol[idx[s]] for s in states if np.isfinite(sol[idx[s]])]
    vmin, vmax = min(finite_vals), max(finite_vals)
    import matplotlib.cm as cm
    cmap = cm.YlGn_r  # low cost = dark green, high cost = yellow

    for i in range(rows):
        for j in range(cols):
            if (i, j) in walls or (i, j) not in state_set:
                color = '#2c2c2c'
                label = ''
            else:
                v = sol[idx[(i, j)]]
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                color = cmap(norm(v))
                label = f'{v:.2f}'

            rect = patches.Rectangle(
                (j, rows - 1 - i), 1, 1,
                linewidth=0.5,
                edgecolor='#aaaaaa',
                facecolor=color
            )
            ax.add_patch(rect)

            if label:
                ax.text(
                    j + 0.5, rows - 0.5 - i,
                    label,
                    ha='center', va='center',
                    fontsize=8, color='black'
                )

    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                 ax=ax, label='cost-to-go V(s)')
    plt.title('Value function on maze')
    plt.tight_layout()
    plt.show()
