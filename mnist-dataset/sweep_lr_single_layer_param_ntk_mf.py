# sweep_lr_single_layer_param_ntk_mf.py
import math, argparse, json, random, os
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# -------------------------
# Repro + device
# -------------------------
def set_seed(s=0):
    random.seed(s); np.random.seed(s); th.manual_seed(s)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# -------------------------
# One-layer model with parametrization modes
# -------------------------
class OneLayerNTK(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        # NTK: weights are initialized as N(0, 1/sqrt(d))
        self.w = nn.Parameter(th.randn(d_in) / math.sqrt(d_in))
        self.d = d_in

    def forward(self, x):
        return (x @ self.w)[:, None]  # (N,1)

class OneLayerMF(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        # MF: weights are initialized as N(0, 1), and output is scaled by 1/d
        self.w = nn.Parameter(th.randn(d_in))
        self.d = d_in

    def forward(self, x):
        return (x @ self.w)[:, None] * (1 / self.d)  # (N, 1)

# -------------------------
# Load MNIST
# -------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# -------------------------
# Train for fixed steps with a given LR
# -------------------------
def train_once(dset_tr, dset_va, d_in, lr, steps=1000, bs=256, seed=0, param="ntk"):
    set_seed(seed)
    # Select model based on parameter choice
    if param == "ntk":
        model = OneLayerNTK(d_in).to(device)
    elif param == "mf":
        model = OneLayerMF(d_in).to(device)
    else:
        raise ValueError("param must be 'ntk' or 'mf'")
    
    opt = th.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for t in range(steps):
        xb, yb = next(iter(dset_tr))  # Get a batch
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return loss.item()

# -------------------------
# Auto-build η0 grid (expands right until improvement stalls)
# -------------------------
def build_eta0_grid_auto(dims, alpha, param, start=1e-5, grow=10**0.25, max_eta0=1e2,
                         steps=600, batch=256, seeds=2, patience=2):
    def eval_grid(eta0):
        from statistics import median
        vals = []
        for d in dims:
            lr = eta0 * (d ** alpha)
            dset_tr, dset_va = make_dataset(d_in=d, seed=123, param=param)
            vlist = []
            for s in range(seeds):
                v = train_once(dset_tr, dset_va, d, lr, steps=steps, bs=batch, seed=1000+s, param=param)
                vlist.append(v)
            vals.append(median(vlist))
        return float(np.mean(vals))

    grid, losses, worse = [start], [eval_grid(start)], 0
    while grid[-1] < max_eta0 and worse < patience:
        nxt = grid[-1] * grow
        l = eval_grid(nxt)
        grid.append(nxt); losses.append(l)
        worse = worse + 1 if l >= losses[-2] else 0
    return np.array(grid)

# -------------------------
# Sweep LR across feature dims with transfer rule lr = η0 * d^α
# -------------------------
def sweep(dims, base_etas, alpha, steps, bs, seeds, param):
    results = []
    dset_cache = {d: make_dataset(d_in=d, seed=123, param=param) for d in dims}
    for d in dims:
        dset_tr, dset_va = dset_cache[d]
        for eta0 in base_etas:
            lr_raw = eta0 * (d ** alpha)
            vals = []
            for s in range(seeds):
                v = train_once(dset_tr, dset_va, d, lr_raw, steps=steps, bs=bs, seed=1000+s, param=param)
                vals.append(v)
            results.append({
                "param": param,
                "dim": int(d),
                "eta0": float(eta0),
                "lr_raw": float(lr_raw),
                "val_loss": float(np.mean(vals)),
                "val_std": float(np.std(vals)),
            })
    return results

# -------------------------
# Collapse score: lower = better alignment across dims at each η0
# -------------------------
def collapse_score(results):
    eta0s = sorted({r["eta0"] for r in results})
    var_list = []
    for e in eta0s:
        losses = [r["val_loss"] for r in results if abs(r["eta0"] - e) < 1e-18]
        if len(losses) >= 2:
            var_list.append(np.var(losses))
    return float(np.mean(var_list)) if var_list else float("inf")

# -------------------------
# Plotting
# -------------------------
def _savefig(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_results(results, dims, alpha, outdir, param):
    outdir = Path(outdir)

    # (A) Loss vs raw LR
    fig = plt.figure()
    for d in dims:
        xs = [r["lr_raw"] for r in results if r["dim"] == d]
        ys = [r["val_loss"] for r in results if r["dim"] == d]
        o = np.argsort(xs); xs = np.array(xs)[o]; ys = np.array(ys)[o]
        plt.plot(xs, ys, marker="o", label=f"d={d}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("learning rate (raw)")
    plt.ylabel("validation MSE")
    plt.title(f"[{param.upper()}] Loss vs LR  (lr = η₀ · d^{alpha})")
    plt.grid(True, which="both", ls=":"); plt.legend()
    _savefig(fig, outdir, f"{param}_loss_vs_lr_alpha{alpha}")

    # (B) Loss vs base-eta (collapse test)
    fig = plt.figure()
    for d in dims:
        xs = [r["eta0"] for r in results if r["dim"] == d]
        ys = [r["val_loss"] for r in results if r["dim"] == d]
        o = np.argsort(xs); xs = np.array(xs)[o]; ys = np.array(ys)[o]
        plt.plot(xs, ys, marker="o", label=f"d={d}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("base learning rate η₀")
    plt.ylabel("validation MSE")
    plt.title(f"[{param.upper()}] Loss vs η₀ (collapse if α is right)")
    plt.grid(True, which="both", ls=":"); plt.legend()
    _savefig(fig, outdir, f"{param}_loss_vs_eta0_alpha{alpha}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", type=str, default="ntk", choices=["ntk", "mf"],
                    help="Use 'ntk' for Neuro Tangent Kernel, 'mf' for Mean-Field Theory")
    ap.add_argument("--dims", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)

    # Transfer rule
    ap.add_argument("--alpha", type=float, default=None, help="lr = η0 * d^alpha; defaults to 0 for ntk, 1 for mf")
    ap.add_argument("--alpha_search", type=int, default=0, help="1=search alpha grid")
    ap.add_argument("--alpha_min", type=float, default=0.6)
    ap.add_argument("--alpha_max", type=float, default=1.2)
    ap.add_argument("--alpha_points", type=int, default=13)

    # η0 grid
    ap.add_argument("--auto_grid", type=int, default=1, help="1=auto-extend η0 to the right")
    ap.add_argument("--eta0_min", type=float, default=1e-5)
    ap.add_argument("--eta0_max", type=float, default=5.0)
    ap.add_argument("--eta0_points", type=int, default=35)

    # Auto-grid knobs
    ap.add_argument("--auto_start", type=float, default=1e-5)
    ap.add_argument("--auto_grow", type=float, default=10**0.25)  # ~1.78x
    ap.add_argument("--auto_max", type=float, default=1e2)
    ap.add_argument("--auto_steps", type=int, default=600)
    ap.add_argument("--auto_seeds", type=int, default=2)
    ap.add_argument("--auto_patience", type=int, default=2)

    # Misc
    ap.add_argument("--outdir", type=str, default="out_one_layer_param")
    args = ap.parse_args()

    set_seed(0)

    # Set a sensible default alpha per regime if not provided
    if args.alpha is None:
        args.alpha = 0.0 if args.param == "ntk" else 1.0
    print(f"[setup] param={args.param}, expected alpha≈{0 if args.param=='ntk' else 1}, using alpha={args.alpha}")

    # Build eta0 grid
    if args.auto_grid:
        base_etas = build_eta0_grid_auto(
            dims=args.dims, alpha=args.alpha, param=args.param,
            start=args.auto_start, grow=args.auto_grow, max_eta0=args.auto_max,
            steps=args.auto_steps, seeds=args.auto_seeds, patience=args.auto_patience
        )
        print(f"[auto-grid] η0 grid has {len(base_etas)} points, max={base_etas[-1]:.4g}")
    else:
        base_etas = np.logspace(
            math.log10(args.eta0_min),
            math.log10(args.eta0_max),
            args.eta0_points
        )
        print(f"[fixed-grid] η0 from {args.eta0_min:g} to {args.eta0_max:g} in {args.eta0_points} points")

    # Single alpha
    if not args.alpha_search:
        results = sweep(args.dims, base_etas, args.alpha, args.steps, args.batch, args.seeds, param=args.param)
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.outdir) / f"{args.param}_results_alpha{args.alpha}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Collapse score: {collapse_score(results):.4e}")
        plot_results(results, args.dims, args.alpha, args.outdir, param=args.param)
        print("Done.")
        return

    # Alpha search
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    best_alpha, best_score, best_results = None, float("inf"), None
    for a in alphas:
        print(f"[alpha-search] α={a:.3f}")
        results = sweep(args.dims, base_etas, a, args.steps, args.batch, args.seeds, param=args.param)
        score = collapse_score(results)
        print(f"  score={score:.4e}")
        if score < best_score:
            best_alpha, best_score, best_results = a, score, results

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir) / f"{args.param}_results_alpha{best_alpha:.3f}.json", "w") as f:
        json.dump(best_results, f, indent=2)
    print(f"[alpha-search] best α={best_alpha:.3f} (score={best_score:.4e})")
    plot_results(best_results, args.dims, best_alpha, args.outdir, param=args.param)
    print("Done.")

if __name__ == "__main__":
    main()
