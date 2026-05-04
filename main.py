# main.py
import random

from maze import make_random_maze
from model_based import make_transition_matrix, make_expcost_matrix, z_iteration
from model_free import ZLearning, QLearning, MazeEnvironmentModelFree
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    n = 10 # Change n here (e.g. 7, 10, 15)
    rows = cols = n
    states, adj, walls = make_random_maze(rows, cols, wall_prob=0.2, seed=42)
    absorbing = random.choices(states, k=2)
    env = MazeEnvironmentModelFree(rows, cols, walls, absorbing)

    # Ground Truth for Error Plot
    P = make_transition_matrix(states, adj)
    G = make_expcost_matrix(states, 1.0, absorbing)
    v_true = -np.log(z_iteration(G, P))
    v_true_norm = (v_true - v_true.min()) / (v_true.max() - v_true.min() + 1e-10)

    total_samples = max(5000, int((n**2) * 150))
    all_samples = env.generate_samples(total_samples)
    sample_steps = np.linspace(100, total_samples, 20, dtype=int)
    err_z, err_q = [], []

    for step in sample_steps:
        z_l, q_l = ZLearning(env.n_states), QLearning(env.n_states, env)
        for k, (s, sn, r, a) in enumerate(all_samples[:step]):
            z_l.update(s, sn, r, k)
            q_l.update(s, sn, r, a, k)
        vz_n = (z_l.get_values() - z_l.get_values().min()) / (z_l.get_values().max() - z_l.get_values().min() + 1e-10)
        vq_val = q_l.get_values()
        vq_n = (vq_val - vq_val.min()) / (vq_val.max() - vq_val.min() + 1e-10)
        err_z.append(np.mean((vz_n - v_true_norm)**2))
        err_q.append(np.mean((vq_n - v_true_norm)**2))

    # Final results for maze plots
    z_fin, q_fin = ZLearning(env.n_states), QLearning(env.n_states, env)
    for k, (s, sn, r, a) in enumerate(all_samples):
        z_fin.update(s, sn, r, k)
        q_fin.update(s, sn, r, a, k)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def plot_maze(ax, values, title):
        grid = np.full((rows, cols), np.nan)
        
        for s, i in env.state_idx.items(): 
            grid[s] = values[i]
        
        ax.set_facecolor('#222222') # Gris très foncé/Noir
        
        im = ax.imshow(grid, cmap='viridis_r', origin='upper', interpolation='none')
        
        ax.set_title(title, fontweight='bold')
        
        for r, c in absorbing: 
            ax.text(c, r, '*', color='red', ha='center', va='center', 
                    fontsize=20, fontweight='bold')
        
        ax.axis('off')
        plt.colorbar(im, ax=ax, label="Normalized Cost")

    plot_maze(axes[0], z_fin.get_values(), 'Z-Learning (Cost-to-go)')
    plot_maze(axes[1], q_fin.get_values(), 'Q-Learning (Cost-to-go)')

    axes[2].plot(sample_steps, err_z, label='Z-Learning', marker='o')
    axes[2].plot(sample_steps, err_q, label='Q-Learning', marker='s')
    axes[2].set_title('Approximation Error Trace', fontweight='bold')
    axes[2].set_xlabel('Samples'); axes[2].set_ylabel('MSE (Normalized)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

if __name__ == "__main__": main()