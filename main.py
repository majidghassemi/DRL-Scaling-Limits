import math
import argparse
import json
import random
import os
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Configuration & Utils
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    env_name: str = "CartPole-v1"
    regime: str = "mup"             # 'ntk', 'mft', or 'mup'
    
    # Scaling Parameters
    width: int = 64
    base_lr: float = 0.01           # eta_0
    alpha: float = 1.0              # 0.0 for NTK, 1.0 for MFT/MuP
    
    # "Physics Fix" Parameters (Only applied in 'mup' regime)
    readout_lr_mult: float = 10.0   # Readout moves faster to track Q-values
    grad_clip: float = 1.0          # Critical for stability during initialization shock
    muP_readout_scale: float = 10.0 # Multiplier for output to match Reward Scale

    # Training Params
    steps: int = 20000
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

def get_device():
    return th.device("cuda" if th.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 2. Replay Buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------------------------
# 3. Scaled MLP (The Physics Engine)
# -----------------------------------------------------------------------------

class ScaledMLP(nn.Module):
    """
    Unified MLP supporting NTK, MFT, and MuP regimes with strict isolation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, config: ExperimentConfig):
        super().__init__()
        self.regime = config.regime.lower()
        self.hidden_dim = hidden_dim
        self.output_mult = config.muP_readout_scale
        
        # Layers
        self.W0 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        # 1. Input Weights (W0): Always He Init (std ~ 1/sqrt(d_in))
        nn.init.kaiming_normal_(self.W0.weight, a=0, mode='fan_in', nonlinearity='relu')

        # 2. Readout Weights (W1) - The Critical Difference
        if self.regime == 'ntk':
            # NTK (Lazy): Standard Normal. Large noise.
            nn.init.normal_(self.W1.weight, mean=0.0, std=1.0)
            
        elif self.regime == 'mft':
            # MFT (Standard): Standard Normal. Large noise causes Init Mismatch.
            nn.init.normal_(self.W1.weight, mean=0.0, std=1.0)
            
        elif self.regime == 'mup':
            # MuP (Ours): Zero Init. Removes noise, enables safe alignment.
            nn.init.zeros_(self.W1.weight)
        else:
            raise ValueError(f"Unknown regime: {self.regime}")

    def forward(self, x):
        h = self.act(self.W0(x))
        
        if self.regime == 'ntk':
            # NTK Scaling: 1 / sqrt(N)
            out = self.W1(h) / math.sqrt(self.hidden_dim)
            
        elif self.regime == 'mft':
            # Standard MFT Scaling: 1 / N (Too small for RL rewards!)
            out = self.W1(h) / self.hidden_dim
            
        elif self.regime == 'mup':
            # MuP Scaling: (1/N) * Multiplier (Matches RL rewards)
            out = (self.W1(h) / self.hidden_dim) * self.output_mult
            
        return out

    def get_hidden_activations(self, x):
        """Used for Rank Analysis"""
        with th.no_grad():
            return self.act(self.W0(x))

# -----------------------------------------------------------------------------
# 4. DQN Agent
# -----------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = get_device()
        self.env = gym.make(config.env_name)
        
        obs_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        
        self.policy_net = ScaledMLP(obs_dim, config.width, n_actions, config).to(self.device)
        self.target_net = ScaledMLP(obs_dim, config.width, n_actions, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # --- OPTIMIZATION SETUP ---
        # Base Rule: eta = eta0 * N^alpha
        base_scaled_lr = config.base_lr * (config.width ** config.alpha)
        
        if config.regime == 'mup':
            # MuP Fix: Split Optimization
            # Readout layer needs to move faster to track Q-values
            readout_lr = base_scaled_lr * config.readout_lr_mult
            params = [
                {'params': self.policy_net.W0.parameters(), 'lr': base_scaled_lr}, # Features
                {'params': self.policy_net.W1.parameters(), 'lr': readout_lr}     # Values
            ]
            self.optimizer = th.optim.SGD(params)
        else:
            # NTK/MFT: Standard Uniform Optimization (Baseline behavior)
            self.optimizer = th.optim.SGD(self.policy_net.parameters(), lr=base_scaled_lr)

        self.memory = ReplayBuffer(capacity=10000)
        self.total_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return self.env.action_space.sample()
        
        with th.no_grad():
            s_t = th.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(s_t)
            return q_vals.argmax().item()

    def train_step(self):
        if len(self.memory) < self.config.batch_size:
            return None

        s, a, r, ns, d = self.memory.sample(self.config.batch_size)
        
        s_b = th.FloatTensor(s).to(self.device)
        a_b = th.LongTensor(a).to(self.device).unsqueeze(1)
        r_b = th.FloatTensor(r).to(self.device).unsqueeze(1)
        ns_b = th.FloatTensor(ns).to(self.device)
        d_b = th.FloatTensor(d).to(self.device).unsqueeze(1)

        with th.no_grad():
            target_q = self.target_net(ns_b).max(1, keepdim=True)[0]
            y = r_b + self.config.gamma * target_q * (1 - d_b)

        current_q = self.policy_net(s_b).gather(1, a_b)
        loss = F.mse_loss(current_q, y)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Stability: Gradient Clipping (Crucial for RL initialization)
        th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.config.grad_clip)
        
        self.optimizer.step()
        return loss.item()

    def compute_effective_rank(self, batch_size=128):
        """
        Calculates Effective Rank. Low rank = Feature Learning.
        """
        if len(self.memory) < batch_size:
            return float(self.config.width)

        s, _, _, _, _ = self.memory.sample(batch_size)
        s_b = th.FloatTensor(s).to(self.device)
        H = self.policy_net.get_hidden_activations(s_b)
        
        try:
            _, S, _ = th.linalg.svd(H)
            p = S / S.sum()
            entropy = -th.sum(p * th.log(p + 1e-9))
            eff_rank = th.exp(entropy).item()
            return eff_rank
        except:
            return float(self.config.width)

    def run_training(self):
        state, _ = self.env.reset(seed=self.config.seed)
        
        episode_rewards = []
        curr_ep_reward = 0
        rank_history = []
        
        # Buffer Warmup
        for _ in range(self.config.batch_size):
            action = self.env.action_space.sample()
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.push(state, action, reward, next_state, done or trunc)
            state = next_state if not (done or trunc) else self.env.reset()[0]
        
        state, _ = self.env.reset(seed=self.config.seed)

        while self.total_steps < self.config.steps:
            # Decaying Epsilon
            epsilon = max(0.05, 1.0 - self.total_steps / 2000)
            
            action = self.select_action(state, epsilon)
            next_state, reward, done, trunc, _ = self.env.step(action)
            done_flag = done or trunc
            
            self.memory.push(state, action, reward, next_state, done_flag)
            state = next_state
            curr_ep_reward += reward
            self.total_steps += 1
            
            if done_flag:
                episode_rewards.append(curr_ep_reward)
                curr_ep_reward = 0
                state, _ = self.env.reset()
            
            self.train_step()
            
            if self.total_steps % 500 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if self.total_steps % 1000 == 0:
                rank_history.append(self.compute_effective_rank())

        avg_ret = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
        final_rank = rank_history[-1] if rank_history else float(self.config.width)
        
        return {
            "return": avg_ret,
            "final_rank": final_rank,
            "rank_history": rank_history
        }

# -----------------------------------------------------------------------------
# 5. Sweep Logic
# -----------------------------------------------------------------------------

def run_sweep(args):
    base_etas = np.logspace(math.log10(args.eta0_min), math.log10(args.eta0_max), args.eta0_points)
    results = []
    print(f"--- STARTING SWEEP: {args.regime.upper()} ---")
    
    for width in args.widths:
        for eta0 in base_etas:
            returns = []
            ranks = []
            for s in range(args.seeds):
                config = ExperimentConfig(
                    env_name=args.env,
                    regime=args.regime,
                    width=width,
                    base_lr=eta0,
                    alpha=args.alpha,
                    steps=args.steps,
                    batch_size=args.batch,
                    seed=1000 + s,
                    readout_lr_mult=args.readout_mult,
                    grad_clip=args.clip
                )
                agent = DQNAgent(config)
                metrics = agent.run_training()
                returns.append(metrics["return"])
                ranks.append(metrics["final_rank"])
            
            mean_ret = np.mean(returns)
            mean_rank = np.mean(ranks)
            lr_actual = eta0 * (width ** args.alpha)
            
            print(f"N={width:<4} | eta0={eta0:.2e} | Ret={mean_ret:>6.1f} | Rank={mean_rank:.1f}")
            results.append({
                "width": width, "eta0": float(eta0), "lr_actual": float(lr_actual),
                "return": float(mean_ret), "effective_rank": float(mean_rank)
            })
    return results

# -----------------------------------------------------------------------------
# 6. Stylized Plotting (The "3+1" Layout)
# -----------------------------------------------------------------------------

def generate_paper_figures():
    """
    Generates high-resolution figures. 
    Stylized with Cool Colors, Large Fonts, NO Legends.
    """
    def load_data(filepath):
        if not Path(filepath).exists():
            print(f"Error: Missing {filepath}.")
            return []
        with open(filepath, 'r') as f:
            return json.load(f)

    print("\n--- Generating Stylized Paper Figures ---")
    ntk_data = load_data("results_ntk/data_ntk.json")
    mft_data = load_data("results_mft/data_mft.json")
    mup_data = load_data("results_mup/data_mup.json")

    if not (ntk_data and mft_data and mup_data):
        print("Skipping plotting: Data missing.")
        return

    # --- STYLE CONFIG ---
    colors = {
        "NTK": "#008B8B",   # DarkCyan
        "MFT": "#4169E1",   # RoyalBlue
        "MuP": "#663399"    # RebeccaPurple
    }
    
    fs_title = 22
    fs_label = 20
    fs_tick = 16
    lw = 3.5
    ms = 8

    # ---------------------------------------------------------
    # FIGURE 1: Scaling Proof (3 Subplots)
    # ---------------------------------------------------------
    fig1, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    regimes = [
        ("NTK (Lazy)", ntk_data, axes[0], colors["NTK"]),
        ("Standard MFT", mft_data, axes[1], colors["MFT"]),
        ("MuP (Ours)", mup_data, axes[2], colors["MuP"])
    ]

    for name, data, ax, color in regimes:
        widths = sorted(list(set(d['width'] for d in data)))
        for i, w in enumerate(widths):
            subset = sorted([d for d in data if d['width'] == w], key=lambda x: x['eta0'])
            xs = [d['eta0'] for d in subset]
            ys = [d['return'] for d in subset]
            
            style = '-' if i == 0 else '--'
            marker = 'o' if i == 0 else 's'
            alpha = 0.8 if i == 0 else 1.0
            
            ax.plot(xs, ys, linestyle=style, marker=marker, color=color, 
                    linewidth=lw, markersize=ms, alpha=alpha)

        ax.set_xscale('log')
        ax.set_title(name, fontsize=fs_title, fontweight='bold', pad=15)
        ax.set_xlabel(r'Base LR $\eta_0$', fontsize=fs_label)
        ax.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax.grid(True, which='both', linestyle=':', alpha=0.5, linewidth=1.5)

    axes[0].set_ylabel('Average Return', fontsize=fs_label)
    plt.tight_layout()
    plt.savefig('fig1_scaling_row.png', dpi=300)
    print("Saved fig1_scaling_row.png")

    # ---------------------------------------------------------
    # FIGURE 2: Rank Comparison (1 Overlay)
    # ---------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(12, 9))
    target_width = 128
    
    datasets = [
        ("NTK", ntk_data, colors["NTK"], ':'),
        ("Standard MFT", mft_data, colors["MFT"], '--'),
        ("MuP (Ours)", mup_data, colors["MuP"], '-')
    ]

    for name, data, color, style in datasets:
        subset = sorted([d for d in data if d['width'] == target_width], key=lambda x: x['eta0'])
        if not subset: continue
        
        xs = [d['eta0'] for d in subset]
        ys = [d['effective_rank'] / d['width'] for d in subset]
        
        ax.plot(xs, ys, color=color, linestyle=style, linewidth=lw+1)

    ax.set_xscale('log')
    ax.set_ylabel('Normalized Effective Rank', fontsize=fs_label)
    ax.set_xlabel(r'Base LR $\eta_0$', fontsize=fs_label)
    ax.set_title(f'Feature Learning Verification (N={target_width})', fontsize=fs_title, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=fs_tick)
    ax.grid(True, which='both', linestyle='-', alpha=0.3, linewidth=1.5)

    plt.tight_layout()
    plt.savefig('fig2_rank_overlay.png', dpi=300)
    print("Saved fig2_rank_overlay.png")

# -----------------------------------------------------------------------------
# 7. Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="train", choices=["train", "plot"])
    parser.add_argument("--regime", type=str, default="mup", choices=['ntk', 'mft', 'mup'])
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta0_min", type=float, default=1e-4)
    parser.add_argument("--eta0_max", type=float, default=1.0)
    parser.add_argument("--eta0_points", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--readout_mult", type=float, default=10.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.action == "plot":
        generate_paper_figures()
    else:
        if args.outdir is None:
            args.outdir = f"results_{args.regime}"
        results = run_sweep(args)
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.outdir) / f"data_{args.regime}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Training complete. Data saved to {args.outdir}")