# main.py
import random
import numpy as np

from maze import make_random_maze
from model_based import make_transition_matrix, make_expcost_matrix, z_iteration
from model_free import ZLearning, QLearning, MazeEnvironmentModelFree
from plotting import show_all_results


def _norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-10)


def compute_error_trace(all_samples, sample_steps, env, absorbing_idx, v_true_norm):
    err_z, err_q = [], []
    for step in sample_steps:
        z_l = ZLearning(env.n_states)
        q_l = QLearning(env.n_states, env)
        for k, (s, sn, r, a) in enumerate(all_samples[:step]):
            z_l.update(s, sn, r, k)
            q_l.update(s, sn, r, a, k)

        v_z = _norm(z_l.get_values())
        v_q = _norm(q_l.get_values())

        denom = np.max(np.abs(v_true_norm))
        err_z.append(np.max(np.abs(v_z - v_true_norm)) / denom)
        err_q.append(np.max(np.abs(v_q - v_true_norm)) / denom)

    return err_z, err_q


def train_final_agents(all_samples, env, absorbing_idx):
    """Train Z- and Q-learning on the full sample budget."""
    z_fin = ZLearning(env.n_states)
    q_fin = QLearning(env.n_states, env)
    for k, (s, sn, r, a) in enumerate(all_samples):
        z_fin.update(s, sn, r, k)
        q_fin.update(s, sn, r, a, k)
    return z_fin, q_fin


def main():
    n = 15        # grid side length — try 7, 10, 15
    rows = cols = n

    states, adj, walls = make_random_maze(rows, cols, wall_prob=0.2, seed=42)
    absorbing     = random.sample(states, k=2)   # sample avoids duplicates
    env           = MazeEnvironmentModelFree(rows, cols, walls, absorbing)
    absorbing_idx = [env.state_idx[a] for a in absorbing]

    # ── Ground truth (model-based z-iteration) ────────────────────────
    P      = make_transition_matrix(states, adj)
    G      = make_expcost_matrix(states, 1.0, absorbing)
    z_gt   = z_iteration(G, P)
    v_true = -np.log(np.clip(z_gt, 1e-12, None))

    # ── Model-free training ───────────────────────────────────────────
    total_samples = max(5000, int((n ** 2) * 150))
    all_samples   = env.generate_samples(total_samples)
    sample_steps  = np.linspace(100, total_samples, 20, dtype=int)

    err_z, err_q = compute_error_trace(all_samples, sample_steps, env, absorbing_idx, _norm(v_true))
    z_fin, q_fin = train_final_agents(all_samples, env, absorbing_idx)

    # ── Visualise ─────────────────────────────────────────────────────
    show_all_results(
        v_true       = v_true,
        v_z          = z_fin.get_values(),
        v_q          = q_fin.get_values(),
        sample_steps = sample_steps,
        err_z        = err_z,
        err_q        = err_q,
        state_idx    = env.state_idx,
        rows         = rows,
        cols         = cols,
        absorbing    = absorbing,
    )


if __name__ == '__main__':
    main()