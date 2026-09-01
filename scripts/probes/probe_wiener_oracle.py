#!/usr/bin/env python
"""Controlled Gaussian validation of the Dual-Wiener route solver.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
It does NOT validate that the plug-in estimator recovers the true SNR of real
data -- nothing can, because the true conditional gradient signal is not
observable.  What it does is construct a world in which the paper's stated model
is TRUE BY CONSTRUCTION, and then ask whether the shipped code returns the
correct answer in that world.

Concretely: freeze a checkpoint, declare a known output-space signal covector
``b`` and a known Gaussian noise covector ``eta ~ N(0, sigma^2)``, and push each
through the REAL computation graph.  Because reverse mode is linear in the
incoming covector, this yields the exact route decomposition

    d_I = s_I + n_I ,   d_J = s_J + n_J

with ``s`` and ``n`` separately known.  Everything the estimand needs is then
exact, so the optimal ``(alpha, m)`` is exact, and any gap is the code's.

LEVEL A -- estimand and solver
    Harvests the four route covectors through a hook on the controller's own
    ``_record_route``, so the measurement point is literally the one the
    controller uses.  Builds the 4x4 Gram of ``(s_I, s_J, n_I, n_J)``, from which
    P, R, the cross-term (a Monte-Carlo zero check) and the empirical risk of ANY
    coefficient pair follow in closed form.  Then compares:

        open     (1, 1)                       exact BPTT at this merge
        oracle   solve_box_wiener_2x2(P+R, R) the shipped solver on exact moments
        grid     brute-force argmin           independent check of the solver
        ckpt     the trained coefficients     how far training landed
        tied     alpha = m = c, c optimal     does the SECOND coefficient pay?
        indep    per-route scalars            does the CROSS term matter?

    ``tied`` is the control the paper says it has not run.  This does not settle
    it -- it settles the estimand-level question (is a tied scalar suboptimal for
    the routing risk?), not the training-outcome question (does that gap change
    forecasting).  Report it as the former.

LEVEL B -- the estimator end to end   (--pipeline N)
    Drives the controller's real calibration path (begin_batch / probe_terms /
    set_probe_losses / calibrate / end_batch) on synthetic batches whose residual
    is exactly ``b + eta``, and compares where it converges against the Level-A
    oracle.  Two sigma sources:

        --sigma-source true    residual buffers forced to the true sigma.
                               Isolates the solver + probe plumbing.
        --sigma-source plugin  the lagged residual EMA runs as in training.
                               Its variance is var(b) + sigma^2, not sigma^2, so
                               this MEASURES the plug-in bias the paper admits
                               qualitatively.  The gap between the two runs is
                               the size of that bias, in coefficient units.

CAVEAT worth carrying into the text: the 2x2 estimand covers the two TOKEN
routes at a merge.  ``branch_state_input`` also applies ``m`` to the recurrent
state inputs, and that route is not part of the solved problem.  This probe
measures what the solver solves, not that extra reuse.

Usage
-----
  python scripts/probes/probe_wiener_oracle.py \
      --ckpt experiments/dual_wiener_screen/mackey_glass/tau30_K32/dualwiener/seed0/last.pth \
      --npz  data/synthetic/mg_D8_tau30.0_dt1.0_sdt0.1_T2048_traj40_b0.2_g0.1_n10.0_tr1000_s0.npz \
      --state-key trajs --hidden 128 --K 32 --burnin 32 \
      --draws 4 --noise-draws 16 --snr-sweep 0.5,1,2,4 \
      --pipeline 40 --out wiener_oracle_mg.npz

  # NARMA (driven)
  python scripts/probes/probe_wiener_oracle.py \
      --ckpt experiments/dual_wiener_screen/narma/L5_K32/dualwiener/seed0/last.pth \
      --npz  data/synthetic/narma_D8_L5_T2048_traj40_tr200_u0.5_bd1_dr1.5_s0.npz \
      --state-key y --stim-key u --hidden 128 --K 32 --burnin 32 \
      --draws 4 --noise-draws 16 --pipeline 40 --out wiener_oracle_narma.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from probe_setup import add_common_args, burn_in_root, setup  # noqa: E402

from internal_dw.models.dual_wiener import solve_box_wiener_2x2  # noqa: E402

FORMAT_VERSION = 1
VNAMES = ("s_I", "s_J", "n_I", "n_J")
ARMS = ("open", "oracle", "grid", "ckpt", "tied", "indep")


# --------------------------------------------------------------------------
# risk algebra
# --------------------------------------------------------------------------
def risk_from_gram(G: np.ndarray, w: np.ndarray) -> float:
    """Empirical E||W d - s||^2 from the 4x4 Gram of (s_I, s_J, n_I, n_J).

    With u = (alpha-1, m-1, alpha, m) the residual of the routing operator is
    exactly ``sum_a u_a v_a``, so the risk is the quadratic form u^T G u.  No
    orthogonality is assumed here: whatever cross term the finite sample has is
    carried, which is why this is the *empirical* risk.
    """
    u = np.array([w[0] - 1.0, w[1] - 1.0, w[0], w[1]], dtype=np.float64)
    return float(u @ G @ u)


def risk_analytic(P: np.ndarray, R: np.ndarray, w: np.ndarray) -> float:
    """The idealized risk (w-1)^T P (w-1) + w^T R w, i.e. cross term set to 0."""
    d = np.asarray(w, dtype=np.float64) - 1.0
    return float(d @ P @ d + np.asarray(w, dtype=np.float64) @ R @ np.asarray(w, dtype=np.float64))


def grid_argmin(P: np.ndarray, R: np.ndarray, n: int) -> np.ndarray:
    g = np.linspace(0.0, 1.0, int(n))
    A, M = np.meshgrid(g, g, indexing="ij")
    W = np.stack([A.ravel(), M.ravel()], axis=1)          # [n*n, 2]
    C = P + R
    obj = np.einsum("bi,ij,bj->b", W, C, W) - 2.0 * (W @ (P @ np.ones(2)))
    return W[int(np.argmin(obj))]


def tied_optimum(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    one = np.ones(2)
    num = float(one @ P @ one)
    den = float(one @ (P + R) @ one)
    c = 0.0 if den <= 0 else float(np.clip(num / den, 0.0, 1.0))
    return np.array([c, c])


def indep_optimum(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    C = P + R
    out = np.ones(2)
    for i in (0, 1):
        out[i] = 0.0 if C[i, i] <= 0 else float(np.clip(P[i, i] / C[i, i], 0.0, 1.0))
    return out


# --------------------------------------------------------------------------
# harvesting
# --------------------------------------------------------------------------
class RouteCapture:
    """Replace the controller's ``_record_route`` so we keep VECTORS, not moments.

    Instance-attribute assignment is enough: ``_DualWienerRoute.backward`` calls
    ``controller._record_route(...)``, which resolves on the instance first.  The
    controller's own method is restored on ``close()``.
    """

    def __init__(self, dw):
        self.dw = dw
        self._orig = dw._record_route
        self.buf: dict[int, dict[int, torch.Tensor]] = {}
        dw._record_route = self._hook

    def _hook(self, which, route_index, slot, gradient):
        self.buf.setdefault(int(route_index), {})[int(which)] = \
            gradient.reshape(gradient.shape[0], -1).double()

    def reset(self):
        self.buf = {}

    def close(self):
        self.dw._record_route = self._orig


def push(loss, roots, capture: RouteCapture, retain: bool = True):
    """Differentiate ``loss`` to the roots with the gate FULLY OPEN, capturing
    every route covector on the way.  Returns {route_index: (v_I, v_J)}."""
    capture.reset()
    torch.autograd.grad(loss, roots, retain_graph=retain, allow_unused=True)
    out = {}
    for ri, d in capture.buf.items():
        if 0 in d and 1 in d and d[0].shape == d[1].shape:
            out[ri] = (d[0], d[1])
    return out


# --------------------------------------------------------------------------
# main measurement
# --------------------------------------------------------------------------
def rollout(bundle, starts, a, keep_graph=True):
    """K-step rollout from a fresh root.  Returns (preds, roots, targets)."""
    model, xt, ut, rows_t = bundle.model, bundle.xt, bundle.ut, bundle.rows_t
    cur, h, roots, _labels, st = burn_in_root(bundle, starts, a)
    preds, tgts = [], []
    with torch.enable_grad():
        c = cur
        for k in range(a.K):
            stim = ut[rows_t, st + k] if ut is not None else None
            out = model.step(h, c, stim_t=stim, horizon_index=k, total_horizon=a.K)
            pred, h = out[0], out[1]
            preds.append(pred)
            tgts.append(xt[rows_t, st + k + 1])
            c = pred
    return preds, roots, tgts


def covector_loss(preds, vecs):
    """Loss whose d/d pred_k is exactly vecs[k] / (K * numel), i.e. the same
    normalization the controller's own probes use."""
    return torch.stack([(p * v).mean() for p, v in zip(preds, vecs)]).mean()


def sigma_for(b_k: torch.Tensor, snr: float, mode: str) -> torch.Tensor:
    """Per-dim noise std with rms(sigma) = rms(b)/snr."""
    rms = b_k.pow(2).mean().clamp_min(1e-30).sqrt()
    target = rms / max(snr, 1e-12)
    if mode == "iso":
        return torch.full((b_k.shape[1],), float(target), device=b_k.device, dtype=b_k.dtype)
    shape = b_k.std(dim=0).clamp_min(1e-20)
    shape = shape / shape.pow(2).mean().clamp_min(1e-30).sqrt()
    return shape * target


def measure_level_a(bundle, a, snr: float):
    """Return grams[(k,l)] -> 4x4 float64, plus bookkeeping."""
    dw = bundle.dw
    depth = bundle.depth
    dev = a.device

    gram = {}
    count = {}
    dw_mode_backup = dw._mode
    dw_collecting_backup = dw._collecting
    capture = RouteCapture(dw)
    try:
        for di, starts in enumerate(bundle.draw_starts):
            dw.begin_batch()
            # Current DualWienerController.route_pair() inserts the custom
            # _DualWienerRoute capture nodes only while ``_collecting`` is
            # true.  Older controller revisions inserted them on every
            # backward, which is why the original probe forced this false.
            # Keep the graph fully open through ``_mode='total'`` while still
            # enabling the route hooks required by this Level-A audit.
            dw._collecting = True
            dw._mode = "total"                      # forces route_coefficient -> 1
            dw._slot = 0
            preds, roots, tgts = rollout(bundle, starts, a)

            # --- the KNOWN signal covector -------------------------------
            if a.signal == "residual":
                bs = [(p - t).detach() for p, t in zip(preds, tgts)]
            else:
                g = torch.Generator(device="cpu").manual_seed(a.seed + 977 * di)
                bs = []
                for p, t in zip(preds, tgts):
                    r = (p - t).detach()
                    z = torch.randn(r.shape, generator=g).to(dev, r.dtype)
                    z = z / z.pow(2).mean().clamp_min(1e-30).sqrt()
                    bs.append(z * r.pow(2).mean().clamp_min(1e-30).sqrt())
            sigmas = [sigma_for(b, snr, a.sigma_mode) for b in bs]

            sig = push(covector_loss(preds, bs), roots, capture, retain=True)

            gen = torch.Generator(device="cpu").manual_seed(a.seed + 10007 * di)
            for nd in range(a.noise_draws):
                etas = [s.reshape(1, -1) * torch.randn(b.shape, generator=gen).to(dev, b.dtype)
                        for b, s in zip(bs, sigmas)]
                last = (di == len(bundle.draw_starts) - 1) and (nd == a.noise_draws - 1)
                noi = push(covector_loss(preds, etas), roots, capture, retain=not last)
                for ri, (sI, sJ) in sig.items():
                    if ri not in noi:
                        continue
                    nI, nJ = noi[ri]
                    V = torch.stack([sI, sJ, nI, nJ])                 # [4, B, D]
                    V = V.reshape(4, -1)
                    G = (V @ V.t()) / V.shape[1]                      # mean over B*D
                    h, l = divmod(int(ri), depth)
                    key = (h, l)
                    gram[key] = gram.get(key, 0.0) + G.cpu().numpy()
                    count[key] = count.get(key, 0) + 1
                if last:
                    del sig, noi
            print(f"[levelA snr={snr:g}] draw {di + 1}/{len(bundle.draw_starts)} "
                  f"({a.noise_draws} noise draws) routes={len(gram)}", flush=True)
    finally:
        capture.close()
        dw._mode = dw_mode_backup
        dw._collecting = dw_collecting_backup

    for key in gram:
        gram[key] = gram[key] / max(count[key], 1)
    return gram


def report_level_a(gram, dw, a, snr):
    """Per-route arms, then aggregate.  Returns a dict of arrays for the npz."""
    keys = sorted(gram.keys())
    rows = []
    for (h, l) in keys:
        G = gram[(h, l)]
        P = G[0:2, 0:2].copy()
        R = G[2:4, 2:4].copy()
        Cx = G[0:2, 2:4].copy()
        tP, tR = float(np.trace(P)), float(np.trace(R))
        cross = float(np.abs(Cx).max()) / max(np.sqrt(max(tP, 0) * max(tR, 0)), 1e-30)

        oracle = solve_box_wiener_2x2(
            torch.tensor(P + R, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
        ).numpy().astype(np.float64)
        gridw = grid_argmin(P, R, a.grid)
        ckpt = dw.coefficients[h, l].detach().double().cpu().numpy()
        cand = {
            "open": np.ones(2), "oracle": oracle, "grid": gridw,
            "ckpt": ckpt, "tied": tied_optimum(P, R), "indep": indep_optimum(P, R),
        }
        r_emp = {k: risk_from_gram(G, w) for k, w in cand.items()}
        r_ana = {k: risk_analytic(P, R, w) for k, w in cand.items()}
        rows.append(dict(h=h, l=l, P=P, R=R, cross=cross, tR_over_tT=tR / max(tP + tR, 1e-30),
                         cand=cand, r_emp=r_emp, r_ana=r_ana))

    solver_err = max(float(np.abs(r["cand"]["oracle"] - r["cand"]["grid"]).max()) for r in rows)
    print()
    print(f"=== LEVEL A   snr={snr:g}   sigma={a.sigma_mode}   signal={a.signal}   "
          f"{len(rows)} routes ===")
    print(f"solver vs brute-force grid ({a.grid}x{a.grid}): max |d(alpha,m)| = {solver_err:.3e}"
          + ("   <-- solver disagrees with the grid" if solver_err > 2.0 / (a.grid - 1) else "   OK"))
    print(f"cross-term |E<s,n>| / sqrt(tr P tr R): median {np.median([r['cross'] for r in rows]):.3e} "
          f"max {max(r['cross'] for r in rows):.3e}   (Monte-Carlo zero; large = too few draws)")

    print()
    hdr = (f"{'arm':>7s} {'alpha':>15s} {'m':>15s} {'risk/open':>20s} "
           f"{'routes better':>14s}")
    print(hdr)
    print("-" * len(hdr))
    base = np.array([r["r_emp"]["open"] for r in rows])
    for arm in ARMS:
        al = np.array([r["cand"][arm][0] for r in rows])
        mm = np.array([r["cand"][arm][1] for r in rows])
        rr = np.array([r["r_emp"][arm] for r in rows]) / np.maximum(base, 1e-300)
        better = int((rr < 1.0 - 1e-9).sum())
        print(f"{arm:>7s} {al.mean():7.4f}+-{al.std():5.4f} {mm.mean():7.4f}+-{mm.std():5.4f} "
              f"{np.median(rr):9.5f} [{rr.min():.4f},{rr.max():.4f}] {better:6d}/{len(rows)}")

    ro = np.array([r["r_emp"]["oracle"] for r in rows])
    rt = np.array([r["r_emp"]["tied"] for r in rows])
    ri = np.array([r["r_emp"]["indep"] for r in rows])
    rc = np.array([r["r_emp"]["ckpt"] for r in rows])
    excess_t = (rt - ro) / np.maximum(ro, 1e-300)
    excess_i = (ri - ro) / np.maximum(ro, 1e-300)
    excess_c = (rc - ro) / np.maximum(ro, 1e-300)
    print()
    print("excess risk over the oracle (this is the estimand-level cost of each restriction)")
    for nm, ex in (("tied  alpha=m", excess_t), ("indep no-cross", excess_i),
                   ("ckpt  trained", excess_c)):
        print(f"  {nm:>16s}  median {np.median(ex):+9.4%}   p90 {np.percentile(ex, 90):+9.4%}   "
              f"max {ex.max():+9.4%}   >1% on {int((ex > 0.01).sum())}/{len(ex)} routes")
    print("  NOTE: this is risk under the estimand, NOT a forecasting result.")

    print()
    print("by horizon (mean over layers)")
    hs = sorted({r["h"] for r in rows})
    hdr2 = (f"{'k':>4s} {'a_oracle':>9s} {'m_oracle':>9s} {'a_ckpt':>8s} {'m_ckpt':>8s} "
            f"{'trR/trT':>8s} {'tied excess':>12s}")
    print(hdr2)
    print("-" * len(hdr2))
    for h in hs:
        sel = [r for r in rows if r["h"] == h]
        if not (h < 3 or h >= max(hs) - 2 or h % max(1, len(hs) // 8) == 0):
            continue
        print(f"{h:4d} {np.mean([r['cand']['oracle'][0] for r in sel]):9.4f} "
              f"{np.mean([r['cand']['oracle'][1] for r in sel]):9.4f} "
              f"{np.mean([r['cand']['ckpt'][0] for r in sel]):8.4f} "
              f"{np.mean([r['cand']['ckpt'][1] for r in sel]):8.4f} "
              f"{np.mean([r['tR_over_tT'] for r in sel]):8.4f} "
              f"{np.mean([(r['r_emp']['tied'] - r['r_emp']['oracle']) / max(r['r_emp']['oracle'], 1e-300) for r in sel]):11.4%}")

    out = {
        f"snr{snr:g}__h": np.array([r["h"] for r in rows]),
        f"snr{snr:g}__l": np.array([r["l"] for r in rows]),
        f"snr{snr:g}__P": np.stack([r["P"] for r in rows]),
        f"snr{snr:g}__R": np.stack([r["R"] for r in rows]),
        f"snr{snr:g}__cross": np.array([r["cross"] for r in rows]),
        f"snr{snr:g}__solver_grid_err": np.array([solver_err]),
    }
    for arm in ARMS:
        out[f"snr{snr:g}__w_{arm}"] = np.stack([r["cand"][arm] for r in rows])
        out[f"snr{snr:g}__risk_{arm}"] = np.array([r["r_emp"][arm] for r in rows])
        out[f"snr{snr:g}__riskana_{arm}"] = np.array([r["r_ana"][arm] for r in rows])
    return out, rows


# --------------------------------------------------------------------------
# LEVEL B: drive the real calibration path
# --------------------------------------------------------------------------
def run_pipeline(bundle, a, snr: float, sigma_source: str, oracle_rows):
    dw = bundle.dw
    depth = bundle.depth
    dev = a.device
    start0 = bundle.draw_starts[0]

    saved = {k: v.detach().clone() for k, v in dw.state_dict().items()}
    gen = torch.Generator(device="cpu").manual_seed(a.seed + 31337)
    traj = []
    n_skipped = 0
    solved0 = int(dw.solved_batches.detach().cpu())
    solved = 0
    try:
        if a.pipeline_reset_buffers:
            # Two EMAs carry the checkpoint's real-data statistics into this run:
            # the residual EMA (0.99) and the route-moment EMA (0.95).  Leaving
            # either in place makes a short pipeline a mixture of real and
            # synthetic statistics.  Clearing both, and the update counters that
            # gate the first-write branch, starts the estimator from nothing.
            dw.residual_mean.zero_()
            dw.residual_second.zero_()
            dw.residual_updates.zero_()
            dw.total_moments.zero_()
            dw.noise_moments.zero_()
            dw.total_updates.zero_()
            dw.noise_updates.zero_()
            dw.coefficients.fill_(1.0)
            # Structured/bootstrap estimators carry one lagged residual draw
            # in addition to the scalar residual moments.  Clear it as well so
            # a candidate sweep cannot inherit a template from its checkpoint.
            if hasattr(dw, "residual_template"):
                dw.residual_template.zero_()
            if hasattr(dw, "residual_template_valid"):
                dw.residual_template_valid.zero_()
            if hasattr(dw, "_template_ready_mask"):
                dw._template_ready_mask[:] = False
            if hasattr(dw, "_pending_residual_sample"):
                dw._pending_residual_sample = {}
            n_res = 1.0 - dw.residual_ema ** max(a.pipeline, 1)
            n_rte = 1.0 - dw.ema ** max(a.pipeline, 1)
            print(f"[levelB {sigma_source}] buffers cleared; after {a.pipeline} batches the "
                  f"residual EMA has converged {100*n_res:.1f}% and the route EMA "
                  f"{100*n_rte:.1f}%", flush=True)
            if sigma_source == "plugin" and n_res < 0.99:
                print(f"[levelB {sigma_source}] WARNING: residual EMA only {100*n_res:.1f}% "
                      f"converged -- plug-in magnitudes are not quotable below ~99%. "
                      f"Need --pipeline >= {int(np.ceil(np.log(0.01)/np.log(dw.residual_ema)))}.",
                      flush=True)
        for it in range(a.pipeline):
            starts = bundle.draw_starts[it % len(bundle.draw_starts)] if a.pipeline_vary_start else start0
            dw.begin_batch()
            dw._collecting = True                       # force every batch to probe
            dw._slot = 0
            preds, roots, tgts = rollout(bundle, starts, a)

            bs = [(p - t).detach() for p, t in zip(preds, tgts)]
            sigmas = [sigma_for(b, snr, a.sigma_mode) for b in bs]
            if sigma_source == "true":
                # Force noise_std() to return the TRUE sigma: var = second - mean^2.
                for k in range(a.K):
                    dw.residual_mean[k].zero_()
                    dw.residual_second[k].copy_(sigmas[k].to(dw.residual_second.dtype) ** 2)
                    dw.residual_updates[k].fill_(1)

            totals, noises = [], []
            for k, (p, b, s) in enumerate(zip(preds, bs, sigmas)):
                eta = s.reshape(1, -1) * torch.randn(b.shape, generator=gen).to(dev, b.dtype)
                target = (p.detach() - (b + eta))       # so residual == b + eta exactly
                t_term, n_term = dw.probe_terms(p, target, k)
                if t_term is not None:
                    totals.append(t_term)
                    noises.append(n_term)
            if not totals:
                # Expected on the first batch after --pipeline-reset-buffers:
                # noise_std() has no residual statistics yet, so there is nothing
                # to probe with.  end_batch() flushes the residuals collected by
                # observe_residual, and the next batch proceeds normally.
                n_skipped += 1
                dw.end_batch()
                traj.append(dw.coefficients[: a.K].detach().float().cpu().numpy().copy())
                continue
            dw.set_probe_losses(torch.stack(totals).mean(), torch.stack(noises).mean())
            dw.calibrate()
            if sigma_source == "true":
                dw._pending_residual = {}               # do not let b contaminate the buffers
            dw.end_batch()

            co = dw.coefficients[: a.K].detach().float().cpu().numpy()
            traj.append(co.copy())
            if (it + 1) % max(1, a.pipeline // 8) == 0:
                print(f"[levelB {sigma_source}] batch {it + 1}/{a.pipeline} "
                      f"alpha_mean={co[..., 0].mean():.4f} m_mean={co[..., 1].mean():.4f}",
                      flush=True)
        final = dw.coefficients[: a.K].detach().float().cpu().numpy()
        solved = int(dw.solved_batches.detach().cpu()) - solved0
    finally:
        dw.load_state_dict(saved)
    if solved == 0:
        raise SystemExit("no batch ever calibrated -- residual buffers stayed empty; raise "
                         "--pipeline or drop --pipeline-reset-buffers")

    orc = np.ones((a.K, depth, 2))
    for r in oracle_rows:
        orc[r["h"], r["l"]] = r["cand"]["oracle"]
    err = np.abs(final - orc)
    print()
    print(f"=== LEVEL B   sigma-source={sigma_source}   {a.pipeline} batches, snr={snr:g} ===")
    print(f"  calibrated on {solved}/{a.pipeline} batches ({n_skipped} skipped for want of "
          f"residual statistics); start rotation "
          f"{'ON' if a.pipeline_vary_start else 'OFF'}, buffers "
          f"{'CLEARED' if a.pipeline_reset_buffers else 'INHERITED from the checkpoint'}")
    print(f"  converged   alpha {final[..., 0].mean():.4f}   m {final[..., 1].mean():.4f}")
    print(f"  oracle      alpha {orc[..., 0].mean():.4f}   m {orc[..., 1].mean():.4f}")
    print(f"  |pipeline - oracle|   alpha  mean {err[..., 0].mean():.4f}  max {err[..., 0].max():.4f}")
    print(f"                        m      mean {err[..., 1].mean():.4f}  max {err[..., 1].max():.4f}")
    signed = final - orc
    print(f"  signed bias (neg = over-attenuates)   alpha {signed[..., 0].mean():+.4f}   "
          f"m {signed[..., 1].mean():+.4f}")
    return final, np.stack(traj)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", default="wiener_oracle.npz")
    ap.add_argument("--signal", default="residual", choices=("residual", "random"),
                    help="residual: b = the real prediction residual (realistic direction and "
                         "scale, deterministic given the batch).  random: a fixed Gaussian "
                         "direction matched in RMS -- checks the result is not an artefact of "
                         "b lying in a special subspace.")
    ap.add_argument("--snr", type=float, default=1.0, help="rms(b)/rms(eta)")
    ap.add_argument("--snr-sweep", default="", help="comma list; overrides --snr")
    ap.add_argument("--sigma-mode", default="iso", choices=("iso", "diag"))
    ap.add_argument("--noise-draws", type=int, default=16,
                    help="eta draws per rollout; the cross-term diagnostic shows if this is enough")
    ap.add_argument("--grid", type=int, default=201, help="brute-force grid per axis")
    ap.add_argument("--pipeline", type=int, default=0,
                    help="LEVEL B: run N synthetic calibration batches (0 = skip)")
    ap.add_argument("--sigma-source", default="both", choices=("true", "plugin", "both"))
    ap.add_argument("--pipeline-vary-start", action="store_true",
                    help="cycle the start set across pipeline batches (pooled over states)")
    ap.add_argument("--pipeline-reset-buffers", action="store_true",
                    help="zero BOTH EMAs (residual 0.99 and route 0.95) and the update counters "
                         "before the run, so the pipeline starts from nothing instead of "
                         "inheriting the checkpoint's real-data statistics.  Without this, a "
                         "short run mixes real and synthetic moments and its plug-in "
                         "coefficients are directional only.")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    bundle = setup(a)

    snrs = ([float(s) for s in a.snr_sweep.split(",") if s.strip()]
            if a.snr_sweep.strip() else [a.snr])

    table = {}
    rows_by_snr = {}
    for snr in snrs:
        gram = measure_level_a(bundle, a, snr)
        if not gram:
            raise SystemExit("no routes captured -- the model is not running the dualwiener policy")
        out, rows = report_level_a(gram, bundle.dw, a, snr)
        table.update(out)
        rows_by_snr[snr] = rows

    if a.pipeline > 0:
        srcs = ("true", "plugin") if a.sigma_source == "both" else (a.sigma_source,)
        ref_snr = snrs[0]
        for src in srcs:
            final, traj = run_pipeline(bundle, a, ref_snr, src, rows_by_snr[ref_snr])
            table[f"pipeline_{src}_final"] = final
            table[f"pipeline_{src}_traj"] = traj
        if a.sigma_source == "both":
            ft = table["pipeline_true_final"]
            fp = table["pipeline_plugin_final"]
            print()
            print("=== plug-in bias, isolated ===")
            print(f"  alpha  true-sigma {ft[..., 0].mean():.4f}   plug-in {fp[..., 0].mean():.4f}"
                  f"   shift {fp[..., 0].mean() - ft[..., 0].mean():+.4f}")
            print(f"  m      true-sigma {ft[..., 1].mean():.4f}   plug-in {fp[..., 1].mean():.4f}"
                  f"   shift {fp[..., 1].mean() - ft[..., 1].mean():+.4f}")
            print("  A negative shift is the signal-counted-as-noise bias: the plug-in residual "
                  "variance is var(b)+sigma^2, so R is overstated and the gate closes too far.")

    meta = {"format_version": FORMAT_VERSION, "ckpt": a.ckpt, "axis": bundle.axis,
            "K": a.K, "burnin": a.burnin, "depth": bundle.depth, "batch": bundle.batch,
            "signal": a.signal, "sigma_mode": a.sigma_mode, "snrs": snrs,
            "noise_draws": a.noise_draws, "grid": a.grid, "arms": list(ARMS),
            "pipeline": a.pipeline, "sigma_source": a.sigma_source,
            "noise_model": str(a.dual_wiener_noise_model),
            "state_dim": bundle.state_dim, "stim_dim": bundle.stim_dim,
            "caveat": "2x2 estimand covers the two token routes only; branch_state_input "
                      "reuses m on the recurrent state route, which is not solved for."}
    np.savez(a.out, meta=json.dumps(meta), **table)
    print(f"\n[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
