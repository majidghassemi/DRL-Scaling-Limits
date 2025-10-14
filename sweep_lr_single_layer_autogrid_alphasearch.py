# sweep_lr_single_layer.py
import math, argparse, json, random, os
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# Repro + device
# -------------------------
def set_seed(s=0):
    random.seed(s); np.random.seed(s); th.manual_seed(s)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# -------------------------
# One-layer linear model
# f(x) = (1/sqrt(d)) * w^T x  (keeps output O(1) as d grows)
# -------------------------
class OneLayer(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.w = nn.Parameter(th.randn(d_in) / math.sqrt(d_in))
        self.d = d_in
    def forward(self, x):
        return (x @ self.w)[:, None]  # (N,1)

# -------------------------
# Synthetic teacher data (linear)
# -------------------------
def make_dataset(n_train=4096, n_val=1024, d_in=32, noise=0.0, seed=123):
    rng = np.random.default_rng(seed)
    Xtr = rng.standard_normal((n_train, d_in)).astype(np.float32)
    Xva = rng.standard_normal((n_val,   d_in)).astype(np.float32)
    w_star = rng.standard_normal(d_in).astype(np.float32) / np.sqrt(d_in)
    ytr = (Xtr @ w_star)[:, None]
    yva = (Xva @ w_star)[:, None]
    if noise > 0:
        ytr += noise * rng.standard_normal(ytr.shape).astype(np.float32)
        yva += noise * rng.standard_normal(yva.shape).astype(np.float32)
    return (Xtr, ytr), (Xva, yva)

# -------------------------
# Train for fixed steps with a given LR, report final val loss
# -------------------------
def train_once(dset_tr, dset_va, d_in, lr, steps=1000, bs=256, seed=0):
    set_seed(seed)
    model = OneLayer(d_in).to(device)
    opt = th.optim.SGD(model.parameters(), lr=lr)
    Xtr, ytr = dset_tr; Xva, yva = dset_va

    tr_loader = DataLoader(TensorDataset(th.from_numpy(Xtr), th.from_numpy(ytr)),
                           batch_size=bs, shuffle=True, drop_last=True)
    it = iter(tr_loader)

    model.train()
    for t in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(tr_loader)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Validation loss
    model.eval()
    with th.no_grad():
        va = F.mse_loss(model(th.from_numpy(Xva).to(device)),
                        th.from_numpy(yva).to(device)).item()
    return va

# -------------------------
# Auto-build η0 grid by expanding to the right
# Stops when average (across widths) median loss ceases improving
# -------------------------
def build_eta0_grid_auto(dims, alpha, start=1e-5, grow=10**0.25, max_eta0=1e2,
                         steps=600, batch=256, seeds=2, patience=2):
    def eval_grid(eta0):
        from statistics import median
        vals = []
        for d in dims:
            lr = eta0 * (d ** alpha)
            dset_tr, dset_va = make_dataset(d_in=d, seed=123)
            vlist = []
            for s in range(seeds):
                v = train_once(dset_tr, dset_va, d, lr, steps=steps, bs=batch, seed=1000+s)
                vlist.append(v)
            vals.append(median(vlist))
        return float(np.mean(vals))

    grid, losses, worse = [start], [eval_grid(start)], 0
    while grid[-1] < max_eta0 and worse < patience:
        nxt = grid[-1] * grow
        l = eval_grid(nxt)
        grid.append(nxt); losses.append(l)
        worse = worse + 1 if l >= losses[-2] else 0  # stop when improvement stalls
    return np.array(grid)

# -------------------------
# Sweep LR across feature dims with transfer rule
# -------------------------
def sweep(dims, base_etas, alpha, steps, bs, seeds):
    results = []
    # Cache datasets per dim so all alphas share data
    dset_cache = {d: make_dataset(d_in=d, seed=123) for d in dims}
    for d in dims:
        dset_tr, dset_va = dset_cache[d]
        for eta0 in base_etas:
            lr_raw = eta0 * (d ** alpha)   # transfer rule: lr = eta0 * d^alpha
            vals = []
            for s in range(seeds):
                v = train_once(dset_tr, dset_va, d, lr_raw, steps=steps, bs=bs, seed=1000+s)
                vals.append(v)
            results.append({
                "dim": int(d),
                "eta0": float(eta0),
                "lr_raw": float(lr_raw),
                "val_loss": float(np.mean(vals)),
                "val_std": float(np.std(vals)),
            })
    return results

# -------------------------
# Collapse score: lower = better alignment across widths at each η0
# -------------------------
def collapse_score(results, dims):
    # For each η0, compute variance across dims; average over η0
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
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png.resolve())

def plot_results(results, dims, alpha, outdir):
    outdir = Path(outdir)

    # (A) Loss vs raw LR (matches the left-board idea)
    fig = plt.figure()
    for d in dims:
        xs = [r["lr_raw"] for r in results if r["dim"] == d]
        ys = [r["val_loss"] for r in results if r["dim"] == d]
        order = np.argsort(xs); xs = np.array(xs)[order]; ys = np.array(ys)[order]
        plt.plot(xs, ys, marker="o", label=f"d={d}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("learning rate (raw)")
    plt.ylabel("validation MSE")
    plt.title(f"Loss vs LR across widths  (transfer lr = η₀ · d^{alpha})")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    _savefig(fig, outdir, f"one_layer_loss_vs_lr_alpha{alpha}")

    # (B) Loss vs base-eta (collapse test)
    fig = plt.figure()
    for d in dims:
        xs = [r["eta0"] for r in results if r["dim"] == d]
        ys = [r["val_loss"] for r in results if r["dim"] == d]
        order = np.argsort(xs); xs = np.array(xs)[order]; ys = np.array(ys)[order]
        plt.plot(xs, ys, marker="o", label=f"d={d}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("base learning rate η₀")
    plt.ylabel("validation MSE")
    plt.title(f"Loss vs base-eta (curves collapse if transfer works; α={alpha})")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    _savefig(fig, outdir, f"one_layer_loss_vs_eta0_alpha{alpha}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Experiment scale
    ap.add_argument("--dims", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)

    # Transfer rule
    ap.add_argument("--alpha", type=float, default=1.0, help="lr = η0 * d^alpha")
    ap.add_argument("--alpha_search", type=int, default=0, help="1=search alpha grid")
    ap.add_argument("--alpha_min", type=float, default=0.6)
    ap.add_argument("--alpha_max", type=float, default=1.2)
    ap.add_argument("--alpha_points", type=int, default=13)

    # η0 grid (two modes)
    ap.add_argument("--auto_grid", type=int, default=1, help="1=auto-extend η0 to the right")
    ap.add_argument("--eta0_min", type=float, default=1e-5)
    ap.add_argument("--eta0_max", type=float, default=5.0)
    ap.add_argument("--eta0_points", type=int, default=35)

    # Auto-grid knobs
    ap.add_argument("--auto_start", type=float, default=1e-5)
    ap.add_argument("--auto_grow", type=float, default=10**0.25)  # ~1.778x
    ap.add_argument("--auto_max", type=float, default=1e2)
    ap.add_argument("--auto_steps", type=int, default=600)  # cheaper probe steps
    ap.add_argument("--auto_seeds", type=int, default=2)
    ap.add_argument("--auto_patience", type=int, default=2)

    # Misc
    ap.add_argument("--outdir", type=str, default="out_one_layer")
    args = ap.parse_args()

    set_seed(0)

    # Build eta0 grid
    if args.auto_grid:
        base_etas = build_eta0_grid_auto(
            dims=args.dims,
            alpha=args.alpha,
            start=args.auto_start,
            grow=args.auto_grow,
            max_eta0=args.auto_max,
            steps=args.auto_steps,
            seeds=args.auto_seeds,
            patience=args.auto_patience
        )
        print(f"[auto-grid] η0 grid has {len(base_etas)} points, max={base_etas[-1]:.4g}")
    else:
        base_etas = np.logspace(
            math.log10(args.eta0_min),
            math.log10(args.eta0_max),
            args.eta0_points
        )
        print(f"[fixed-grid] η0 from {args.eta0_min:g} to {args.eta0_max:g} in {args.eta0_points} points")

    # Option A: Single alpha
    if not args.alpha_search:
        results = sweep(args.dims, base_etas, args.alpha, args.steps, args.batch, args.seeds)
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.outdir) / f"results_alpha{args.alpha}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Collapse score (lower is better): {collapse_score(results, args.dims):.4e}")
        plot_results(results, args.dims, args.alpha, args.outdir)
        print("Done.")
        return

    # Option B: Search alpha, pick the best collapse
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    best_alpha, best_score, best_results = None, float("inf"), None
    for a in alphas:
        print(f"[alpha-search] evaluating α={a:.3f}")
        results = sweep(args.dims, base_etas, a, args.steps, args.batch, args.seeds)
        score = collapse_score(results, args.dims)
        print(f"  score={score:.4e}")
        if score < best_score:
            best_alpha, best_score, best_results = a, score, results

    # Save + plot the best
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir) / f"results_alpha{best_alpha:.3f}.json", "w") as f:
        json.dump(best_results, f, indent=2)
    print(f"[alpha-search] best α={best_alpha:.3f}  (score={best_score:.4e})")
    plot_results(best_results, args.dims, best_alpha, args.outdir)
    print("Done.")

if __name__ == "__main__":
    main()
