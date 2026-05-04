
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
class MazeEnvironmentModelFree:
    def __init__(self, rows, cols, walls, absorbing_states, noise=0.1):
        self.rows, self.cols = rows, cols
        self.walls = walls
        self.absorbing_states = absorbing_states
        self.noise = noise
        self.states = [(i, j) for i in range(rows) for j in range(cols) if (i, j) not in walls]
        self.state_idx = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)
        
        def neighbours(i, j):
            candidates = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
            return [(r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < cols and (r, c) not in walls]
        self.adj = {s: neighbours(*s) for s in self.states}

    def generate_samples(self, n_samples):
        samples = []
        non_absorbing = [s for s in self.states if s not in self.absorbing_states]
        while len(samples) < n_samples:
            state = random.choice(non_absorbing)
            # Max 100 steps per trajectory to avoid infinite loops
            for _ in range(100):
                if state in self.absorbing_states: 
                    break
                nbrs = self.adj[state]
                action = random.choice(nbrs)
                # Apply noise
                next_state = action if random.random() < (1.0 - self.noise) else random.choice(nbrs)
                cost = 0.0 if next_state in self.absorbing_states else 1.0
                
                samples.append((self.state_idx[state], self.state_idx[next_state], -cost, self.state_idx[action]))
                state = next_state
        return samples[:n_samples]
# ============================================================================
# ALGORITHMS
# ============================================================================

class ZLearning:
    def __init__(self, n_states):
        self.z_hat = np.ones(n_states)
        self.lr = lambda k: 10.0 / (10.0 + k)
        
    def update(self, s, s_next, reward, k):
        alpha = self.lr(k)
        # ẑ(i) ← (1-α) ẑ(i) + α exp(-cost) ẑ(j)[cite: 1]
        self.z_hat[s] = (1 - alpha) * self.z_hat[s] + alpha * np.exp(reward) * self.z_hat[s_next]
        
    def get_values(self):
        # V = -log(z)[cite: 1]
        return -np.log(np.clip(self.z_hat, 1e-12, None))

class QLearning:
    def __init__(self, n_states, env, gamma=0.95):
        self.n_states, self.env, self.gamma = n_states, env, gamma
        self.Q = defaultdict(lambda: defaultdict(float))
        self.lr = lambda k: 10.0 / (10.0 + k)
        
    def update(self, s, s_next, reward, action, k):
        alpha = self.lr(k)
        next_nbrs = [self.env.state_idx[n] for n in self.env.adj[self.env.states[s_next]]]
        max_q = max([self.Q[s_next][a] for a in next_nbrs]) if next_nbrs and s_next in self.Q else 0.0
        # Q update using negative costs as rewards[cite: 1]
        self.Q[s][action] += alpha * (reward + self.gamma * max_q - self.Q[s][action])
        
    def get_values(self):
        v = np.zeros(self.n_states)
        for i in range(self.n_states):
            nbrs = [self.env.state_idx[n] for n in self.env.adj[self.env.states[i]]]
            # Convert back to a positive "cost-to-go" for visualization consistency
            v[i] = -max([self.Q[i][a] for a in nbrs]) if nbrs and i in self.Q else 0.0
        return v