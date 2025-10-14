# sweep_lr_width_transfer.py
import math, argparse, itertools, json, random, os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# 1) Repro + device
# -------------------------
def set_seed(s=0):
    random.seed(s); np.random.seed(s); th.manual_seed(s)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# -------------------------
# 2) Mean-field 2-layer MLP
# f(x) = (1/n) W1 phi(W0 x)
# -------------------------
class MFTwoLayer(nn.Module):
    def __init__(self, d_in, n_hidden, d_out, act="relu"):
        super().__init__()
        self.d_in, self.n_hidden, self.d_out = d_in, n_hidden, d_out
        self.W0 = nn.Linear(d_in, n_hidden, bias=False)
        self.W1 = nn.Linear(n_hidden, d_out, bias=False)
        self.act = getattr(F, act)
        # He init for W0 to keep preact O(1); W1 ~ N(0,1) then scaled by 1/n in forward
        nn.init.kaiming_normal_(self.W0.weight, nonlinearity=act)
        nn.init.normal_(self.W1.weight, mean=0.0, std=1.0)

    def forward(self, x):
        h = self.act(self.W0(x))
        return (self.W1(h)) / self.n_hidden  # mean-field 1/n scaling

# -------------------------
# 3) Synthetic teacher data (simple but non-trivial)
# -------------------------
def make_dataset(n_train=4096, n_val=1024, d_in=32, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = rng.standard_normal((n_train, d_in)).astype(np.float32)
    Xva = rng.standard_normal((n_val,   d_in)).astype(np.float32)
    # Teacher: fixed wide random net (no training) to give structured targets
    teacher = MFTwoLayer(d_in, 4096, 1).to(device).eval()
    with th.no_grad():
        ytr = teacher(th.from_numpy(Xtr).to(device)).cpu().numpy()
        yva = teacher(th.from_numpy(Xva).to(device)).cpu().numpy()
    ytr = ytr.astype(np.float32); yva = yva.astype(np.float32)
    return (Xtr, ytr), (Xva, yva)

# -------------------------
# 4) Train for fixed steps with a given LR, report final val loss
# -------------------------
def train_once(dset_tr, dset_va, d_in, n_hidden, d_out, lr, steps=1000, bs=256, seed=0):
    set_seed(seed)
    model = MFTwoLayer(d_in, n_hidden, d_out).to(device)
    opt = th.optim.SGD(model.parameters(), lr=lr)  # use SGD to mirror whiteboard intuition
    Xtr, ytr = dset_tr
    Xva, yva = dset_va
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
# 5) Sweep LR across widths with transfer rule
# -------------------------
def sweep(widths, base_etas, alpha, steps, bs, seeds, d_in=32):
    (Xtr, ytr), (Xva, yva) = make_dataset(d_in=d_in, seed=123)
    results = []  # rows: dict(width, lr_raw, eta0, val_loss)
    for n in widths:
        for eta0 in base_etas:
            # Transfer rule: lr_raw = eta0 * n^alpha
            lr_raw = eta0 * (n ** alpha)
            val_losses = []
            for s in range(seeds):
                v = train_once((Xtr, ytr), (Xva, yva),
                               d_in=d_in, n_hidden=n, d_out=1,
                               lr=lr_raw, steps=steps, bs=bs, seed=1000+s)
                val_losses.append(v)
            results.append({
                "width": n,
                "eta0": float(eta0),
                "lr_raw": float(lr_raw),
                "val_loss": float(np.mean(val_losses)),
                "val_std": float(np.std(val_losses))
            })
    return results

# -------------------------
# 6) Plot like the left diagram
# -------------------------
def plot_results(results, widths, base_etas, alpha, outdir="out"):
    os.makedirs(outdir, exist_ok=True)
    # (A) Loss vs raw LR (matches the whiteboard picture)
    plt.figure()
    for n in widths:
        xs = [r["lr_raw"] for r in results if r["width"] == n]
        ys = [r["val_loss"] for r in results if r["width"] == n]
        order = np.argsort(xs); xs = np.array(xs)[order]; ys = np.array(ys)[order]
        plt.plot(xs, ys, marker="o", label=f"n={n}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate (raw)")
    plt.ylabel("validation MSE")
    plt.title(f"Loss vs LR across widths  (transfer lr=eta0 * n^{alpha})")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"loss_vs_lr_raw_alpha{alpha}.png"), dpi=180)

    # (B) Loss vs base-eta (after factoring out n^alpha) to check alignment
    plt.figure()
    for n in widths:
        xs = [r["eta0"] for r in results if r["width"] == n]
        ys = [r["val_loss"] for r in results if r["width"] == n]
        order = np.argsort(xs); xs = np.array(xs)[order]; ys = np.array(ys)[order]
        plt.plot(xs, ys, marker="o", label=f"n={n}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"base learning rate $\eta_0$")
    plt.ylabel("validation MSE")
    plt.title(f"Loss vs base-eta (curves collapse if transfer works; alpha={alpha})")
    plt.legend(); plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"loss_vs_eta0_alpha{alpha}.png"), dpi=180)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--eta0_min", type=float, default=1e-5)
    ap.add_argument("--eta0_max", type=float, default=1.0)
    ap.add_argument("--eta0_points", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0, help="transfer exponent: lr = eta0 * n^alpha")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="out")
    args = ap.parse_args()

    set_seed(0)
    base_etas = np.logspace(math.log10(args.eta0_min), math.log10(args.eta0_max), args.eta0_points)
    results = sweep(args.widths, base_etas, args.alpha, args.steps, args.batch, args.seeds)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f"results_alpha{args.alpha}.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_results(results, args.widths, base_etas, args.alpha, args.outdir)
    print(f"Saved plots to {args.outdir}/")

if __name__ == "__main__":
    main()
