import numpy as np
from scipy.sparse import diags, csr_matrix

# model based example
#rows, cols = 7, 7
#walls = {(2,0),(2,1),(2,2),(2,3),(2,4), (4,2),(4,3),(4,4),(4,5),(4,6)}
#absorbing = {(rows-1, cols-1)}

def make_transition_matrix(states, adj):
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    rows, cols, data = [], [], []
    for s, nbrs in adj.items():
        if nbrs:
            for t in nbrs:
                rows.append(idx[s])
                cols.append(idx[t])
                data.append(1 / len(nbrs))
    return csr_matrix((data, (rows, cols)), shape=(n, n))

def make_expcost_matrix(states, rho, A):
    idx = {s: i for i, s in enumerate(states)}
    diag = [1.0 if s in A else np.exp(-rho) for s in states]
    return diags(diag, format='csr')



def z_iteration(G, P, tol=1e-15, n_iter=100):
    M = G @ P  # sparse @ sparse = sparse, computed once
    z = np.ones(P.shape[0])
    for k in range(n_iter):
        z_new = M @ z  # O(n) since M is sparse
        z_new /= np.linalg.norm(z_new) # unnessary according to the article

        if np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-16) < tol:
           print(f"Converged at iteration {k}")
           break

        z = z_new
    return z
