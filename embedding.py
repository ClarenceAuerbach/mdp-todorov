import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import random
 
from maze import make_random_maze

def _embed_state(B, y):
    """
    For one state with B ∈ R^{m × (m+1)} and y ∈ R^m:
 
    Find  α*  minimising  f(α) = log Σ_j exp(x̂_j + α v_j)
    (log-sum-exp is strictly convex in α → unique global minimum).
 
    Then  q = -f(α*),  p_ij = softmax(x̂ + α* v).
 
    Returns  (p, q)  with p a probability vector of length m+1.
    """
    # Minimum-norm particular solution to  B x = y
    x_hat, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
 
    # 1-D null space of B
    ns = null_space(B)
    if ns.shape[1] == 0:
        v = np.zeros_like(x_hat)
    else:
        v = ns[:, 0]
 
    def lse(alpha):
        x = x_hat + alpha * v
        c = x.max()
        return c + np.log(np.sum(np.exp(x - c)))   # log-sum-exp, stable
 
    if np.linalg.norm(v) > 1e-10:
        res = minimize_scalar(lse, bounds=(-300, 300), method='bounded')
        alpha_opt = res.x
    else:
        alpha_opt = 0.0
 
    x = x_hat + alpha_opt * v
    c = x.max()
    log_Z = c + np.log(np.sum(np.exp(x - c)))
    q = float(-log_Z)
 
    p = np.exp(x - log_Z)          # = softmax(x) = exp(x+q)
    p = np.maximum(p, 0.0)
    p /= p.sum()                    # numerical safety
 
    return p, q
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Full embedding
# ─────────────────────────────────────────────────────────────────────────────
 
def embed_to_continuous(states, adj, costs, trans, absorbing):
    """
    Build the uncontrolled transition matrix P and state-cost vector q
    for the equivalent continuous MDP (eqs. 31–41 of Todorov 2006).
 
    For each non-absorbing state i:
      N(i) = nbrs ∪ {i}  (m+1 possible next states)
      U(i) = nbrs         (m actions)
      B ∈ R^{m × (m+1)}  (wide — has a 1-D null space)
      y[a] = ℓ̃(i,a) − h(i,a)
      (p, q) = _embed_state(B, y)   [null-space optimisation]
 
    Returns:  P (sparse n×n), q_arr (ndarray n,), idx (dict)
    """
    idx = {s: i for i, s in enumerate(states)}
    abs_set = set(absorbing)
    n = len(states)
 
    P = lil_matrix((n, n))
    q_arr = np.zeros(n)
    n_neg = 0
 
    for s in states:
        i = idx[s]
 
        if s in abs_set:
            P[i, i] = 1.0
            q_arr[i] = 0.0
            continue
 
        nbrs = adj[s]
        if not nbrs:
            P[i, i] = 1.0
            continue
 
        N_i = nbrs + [s]           # m+1 possible next states
        m_a = len(nbrs)
        m_n = len(N_i)             # = m_a + 1
 
        # B matrix  (m_a × m_n)
        B = np.zeros((m_a, m_n))
        for a_idx, d in enumerate(trans[s]):
            for j_idx, j in enumerate(N_i):
                B[a_idx, j_idx] = d.get(j, 0.0)
 
        # Action entropy  h(i,a) = −Σ_j p̃_ij(a) log p̃_ij(a)
        h = np.array([
            -sum(p * np.log(p) for p in d.values() if p > 0)
            for d in trans[s]
        ])
 
        y = np.array(costs[s]) - h
 
        p_ij, q_i = _embed_state(B, y)
 
        if q_i < -1e-4:
            n_neg += 1
 
        q_arr[i] = q_i
        for j_idx, j in enumerate(N_i):
            P[i, idx[j]] += float(p_ij[j_idx])
 
    if n_neg:
        print(f"  Note: {n_neg} states still have q(i) < 0; "
              "values may diverge for those states.")
 
    return csr_matrix(P), q_arr, idx
 