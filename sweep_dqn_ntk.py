import math, argparse, json, random, os
from collections import deque
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

# -------------------------
# 1) Setup
# -------------------------
def set_seed(s=0):
    random.seed(s); np.random.seed(s); th.manual_seed(s)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# -------------------------
# 2) NTK Model (Lazy Regime)
# -------------------------
class NTKTwoLayer(nn.Module):
    def __init__(self, d_in, n_hidden, d_out, act="relu"):
        super().__init__()
        self.n_hidden = n_hidden
        self.W0 = nn.Linear(d_in, n_hidden, bias=False)
        self.W1 = nn.Linear(n_hidden, d_out, bias=False)
        self.act = getattr(F, act)
        
        # NTK Init
        # W0: He init (std ~ 1/sqrt(d_in))
        nn.init.kaiming_normal_(self.W0.weight, nonlinearity=act)
        # W1: N(0, 1). We divide by sqrt(N) in forward
        nn.init.normal_(self.W1.weight, mean=0.0, std=1.0) 

    def forward(self, x):
        h = self.act(self.W0(x))
        # NTK Scaling: Divide by sqrt(N)
        return (self.W1(h)) / math.sqrt(self.n_hidden)

# -------------------------
# 3) Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# -------------------------
# 4) RL Loop
# -------------------------
def train_dqn(width, lr, steps=5000, bs=64, seed=0, gamma=0.99):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    
    policy_net = NTKTwoLayer(4, width, 2).to(device)
    target_net = NTKTwoLayer(4, width, 2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    opt = th.optim.SGD(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(capacity=5000)
    
    state, _ = env.reset(seed=seed)
    # Warmup buffer
    for _ in range(bs):
        action = env.action_space.sample()
        next_state, reward, done, trunc, _ = env.step(action)
        memory.push(state, action, reward, next_state, done or trunc)
        state = next_state if not (done or trunc) else env.reset()[0]

    state, _ = env.reset(seed=seed)
    total_steps = 0
    episode_rewards = []
    curr_ep_reward = 0
    
    while total_steps < steps:
        epsilon = max(0.05, 1.0 - total_steps / 1000)
        
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with th.no_grad():
                s_t = th.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = policy_net(s_t)
                action = q_vals.argmax().item()
        
        next_state, reward, done, trunc, _ = env.step(action)
        done_flag = done or trunc
        memory.push(state, action, reward, next_state, done_flag)
        state = next_state
        curr_ep_reward += reward
        total_steps += 1
        
        if done_flag:
            episode_rewards.append(curr_ep_reward)
            curr_ep_reward = 0
            state, _ = env.reset()

        if len(memory) > bs:
            s, a, r, ns, d = memory.sample(bs)
            s_b = th.FloatTensor(s).to(device)
            a_b = th.LongTensor(a).to(device).unsqueeze(1)
            r_b = th.FloatTensor(r).to(device).unsqueeze(1)
            ns_b = th.FloatTensor(ns).to(device)
            d_b = th.FloatTensor(d).to(device).unsqueeze(1)

            with th.no_grad():
                target_q = target_net(ns_b).max(1, keepdim=True)[0]
                y = r_b + gamma * target_q * (1 - d_b)
            
            current_q = policy_net(s_b).gather(1, a_b)
            loss = F.mse_loss(current_q, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if total_steps % 200 == 0:
                target_net.load_state_dict(policy_net.state_dict())

    avg_return = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
    return -avg_return

# -------------------------
# 5) Sweep
# -------------------------
def sweep(widths, base_etas, alpha, steps, bs, seeds):
    results = []
    for n in widths:
        for eta0 in base_etas:
            lr_raw = eta0 * (n ** alpha)
            returns = []
            for s in range(seeds):
                neg_r = train_dqn(width=n, lr=lr_raw, steps=steps, bs=bs, seed=1000+s)
                returns.append(-neg_r)
            
            mean_ret = np.mean(returns)
            print(f"N={n}, eta0={eta0:.2e}, LR={lr_raw:.2e} -> Return={mean_ret:.1f}")
            
            results.append({
                "width": n,
                "eta0": float(eta0),
                "lr_raw": float(lr_raw),
                "val_loss": -float(mean_ret),
                "real_return": float(mean_ret)
            })
    return results

# -------------------------
# 6) Plot
# -------------------------
def plot_results(results, widths, alpha, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure()
    for n in widths:
        xs = [r["eta0"] for r in results if r["width"] == n]
        ys = [r["real_return"] for r in results if r["width"] == n]
        order = np.argsort(xs); xs = np.array(xs)[order]; ys = np.array(ys)[order]
        plt.plot(xs, ys, marker="o", label=f"N={n}")
        
    plt.xscale("log")
    plt.xlabel(r"base learning rate $\eta_0$")
    plt.ylabel("Average Return")
    plt.title(f"DQN NTK Scaling: Return vs Eta0 (alpha={alpha})")
    plt.legend(); plt.grid(True, which="both", ls=":")
    fig.savefig(outdir / f"dqn_ntk_return_vs_eta0_alpha{alpha}.png", bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--eta0_min", type=float, default=1e-3)
    ap.add_argument("--eta0_max", type=float, default=2.0)
    ap.add_argument("--eta0_points", type=int, default=10)
    # Default alpha=0.0 for NTK
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="out_dqn_ntk")
    args = ap.parse_args()

    set_seed(0)
    base_etas = np.logspace(math.log10(args.eta0_min), math.log10(args.eta0_max), args.eta0_points)
    
    print(f"Sweeping DQN NTK | widths={args.widths} | alpha={args.alpha}")
    results = sweep(args.widths, base_etas, args.alpha, args.steps, args.batch, args.seeds)
    
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir) / f"results_dqn_ntk_alpha{args.alpha}.json", "w") as f:
        json.dump(results, f, indent=2)
        
    plot_results(results, args.widths, args.alpha, args.outdir)
    print(f"Done. Saved to {args.outdir}")

if __name__ == "__main__":
    main()