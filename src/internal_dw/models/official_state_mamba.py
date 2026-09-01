"""Official-state Mamba autoregressive model.

This is the *only* Mamba backbone used by the official-shadow patch.  It does
not maintain a rolling token buffer.  The external recurrent state is the real
Mamba inference state for every layer:

    conv_state_l : [B, d_inner, d_conv]
    ssm_state_l  : [B, d_inner, d_state]

The same class supports HCP vector states [B, D] and The Well fields [B, C, H, W]
by flattening frames internally and reshaping predictions back to the reference
frame shape.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_ar import _StimPoolMixin
from .dual_wiener import DualWienerController
from .global_horizon_wiener import GlobalHorizonWienerController


_POL_ANNOUNCED = False   # one-shot: report the routing config the model actually got

MambaLayerState = Tuple[torch.Tensor, torch.Tensor]
MambaStackState = Tuple[MambaLayerState, ...]


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _skip_alpha(horizon_index) -> float:
    """The REACH gate: how much of the temporal carry survives the backward pass.

    Routing gates the nonlinear branch, so the accumulated backward operator is
    prod_k (I + m_k J_k). Every m multiplies a term containing at least one J --
    the length-0 term of that expansion is bare. At m=0 the product is prod I = I:
    the far loss still reaches step 1 at unit gain. Branch routing therefore
    controls AMPLIFICATION and cannot shorten REACH; the identity highway is a
    floor of 1 that no choice of m can lower.

    alpha attaches to the carry itself (the AR skip x_t and the recurrent state),
    giving prod_k (alpha_k I + m_k J_k). The gain is then prod alpha_k, so:

      alpha == 1 everywhere            -> today's ResGrad (no reach control)
      alpha == 0 every S steps, fixed  -> long-forward segment-detached TBPTT,
                                          i.e. the state-INDEPENDENT special case
      alpha state-dependent            -> reach-axis routing (not implemented here)

    A single constant alpha is deliberately NOT offered: prod alpha = alpha^k is a
    function of k alone, which is soft truncation.

      RESGRAD_ALPHA_PERIOD=S   cut the carry every S rollout steps (S<=0 disables)
      RESGRAD_ALPHA_VALUE=v    value at a cut (default 0.0 = a hard detach)
    """
    import os as _os
    raw = _os.environ.get("RESGRAD_ALPHA_PERIOD", "")
    if not raw or horizon_index is None:
        return 1.0
    try:
        period = int(float(raw))
    except ValueError:
        return 1.0
    if period <= 0:
        return 1.0
    if not getattr(_skip_alpha, "_announced", False):
        _skip_alpha._announced = True
        import sys as _sys
        _sys.stderr.write(
            "[reach gate] RESGRAD_ALPHA_PERIOD=%d, alpha at a cut = %s "
            "-> segment-detached TBPTT with the full forward rollout retained\n"
            % (period, _os.environ.get("RESGRAD_ALPHA_VALUE", "0.0")))
        _sys.stderr.flush()
    k = int(horizon_index)
    if k > 0 and (k % period) == 0:
        try:
            return float(_os.environ.get("RESGRAD_ALPHA_VALUE", "0.0"))
        except ValueError:
            return 0.0
    return 1.0


def _apply_alpha(t: torch.Tensor, alpha: float) -> torch.Tensor:
    """Straight-through: forward value unchanged, backward scaled by alpha."""
    if alpha >= 1.0:
        return t
    if alpha <= 0.0:
        return t.detach()
    return t.detach() + alpha * (t - t.detach())


def _corrected_coherence(g: torch.Tensor) -> torch.Tensor:
    """Batch coherence with the finite-sample correction: S/(S+N) per sample.

    Returns a 0-dim tensor ON DEVICE. Never call .cpu()/.item() here: the probe
    fires 2 x depth x layers times per training step, and a device sync at each
    call costs more than the whole backward pass.
    """
    gg = g.reshape(g.shape[0], -1).float()
    B = gg.shape[0]
    num = (gg.mean(dim=0) ** 2).sum()
    den = (gg * gg).sum(dim=1).mean() + 1e-12
    c = num / den
    if B > 1:
        c = (B * c - 1.0) / (B - 1.0)
    return c.clamp(0.0, 1.0).detach()


class _PathProbe(torch.autograd.Function):
    """MEASUREMENT ONLY -- identity in both directions, records nothing forward.

    Deriving alpha and m jointly from g = mu + nu gives, when the paths decouple,
    one Wiener gain per route: alpha* = s_I/(s_I+n_I) and m* = s_J/(s_J+n_J).
    A single coefficient beta(I+J) is the alpha=m diagonal of that family, and it
    is optimal exactly when the two routes have equal SNR. So the question
    "one coefficient or two?" reduces to one measurement: does c_I equal c_J?

    Inside a block the two routes are separable because the input tensor is
    referenced twice -- once as the untouched residual, once through the norm
    into the branch. A probe on each reference sees exactly one route's gradient:

        residual = probe(token, "I")   ->  grad = dL/dout            = g_I
        token    = norm(probe(token,"J")) -> grad = J^T dL/dout      = g_J

    Enable with RESGRAD_MEASURE_DUAL=1 (window via RESGRAD_MEASURE_EVERY).
    """

    cur_k = -1          # set by the stack before each rollout step
    _pending = {}
    _acc = {}           # k -> [sum c_I, sum c_J, n]
    _seen = 0

    @staticmethod
    def forward(ctx, x, which):
        ctx.which = which
        ctx.k = int(_PathProbe.cur_k)   # captured in FORWARD, when it is valid
        return x

    @staticmethod
    def backward(ctx, grad_out):
        _PathProbe._record(ctx.which, ctx.k, grad_out)
        return grad_out, None

    @staticmethod
    def _record(which, k, grad_out):
        if grad_out is None or grad_out.shape[0] < 2:
            return
        _PathProbe._pending[(which, k)] = _corrected_coherence(grad_out.detach())
        if ("I", k) in _PathProbe._pending and ("J", k) in _PathProbe._pending:
            cI = _PathProbe._pending.pop(("I", k))
            cJ = _PathProbe._pending.pop(("J", k))
            a = _PathProbe._acc.get(k)
            if a is None:
                a = [cI, cJ, 1]                 # tensors stay on device
                _PathProbe._acc[k] = a
            else:
                a[0] = a[0] + cI; a[1] = a[1] + cJ; a[2] += 1
            _PathProbe._seen += 1
            import os as _os
            win = int(float(_os.environ.get("RESGRAD_MEASURE_EVERY", "20000") or 20000))
            if _PathProbe._seen >= win:
                _PathProbe._dump()

    @staticmethod
    def _dump():
        import sys as _sys
        ks = sorted(x for x in _PathProbe._acc if x >= 0)
        _sys.stderr.write(
            "\n[G_k]  depth   c_I      c_J      G_k(=c_I)   beta_k=G_k/G_(k-1)   n\n")
        prev = None
        for k in ks:
            sI, sJ, n = _PathProbe._acc[k]
            if n == 0:
                continue
            # the ONLY device sync in the whole measurement
            cI = float(sI.detach().cpu()) / n
            cJ = float(sJ.detach().cpu()) / n
            beta = (cI / prev) if (prev is not None and prev > 1e-12) else float("nan")
            if k in (0, 1, 2, 4, 8, 16, 24, 32, 48, 63) or len(ks) <= 12:
                _sys.stderr.write("[G_k] %6d  %7.5f  %7.5f  %9.5f  %14.4f  %6d\n"
                                  % (k, cI, cJ, cI, beta, n))
            prev = cI
        _sys.stderr.write(
            "[G_k]  beta_k -> 1 at large k means the far credit should NOT be "
            "shrunk further; beta_k << 1 there means it should.\n\n")
        _sys.stderr.flush()
        _PathProbe._acc = {}
        _PathProbe._seen = 0


class _DualGate(torch.autograd.Function):
    """Legacy cross-example-coherence proxy for a two-route backward operator.

    The 2x2 algebra below is exact only after assuming batch rows are IID noisy
    observations of one shared gradient mean.  That assumption is generally
    false when rows represent different states/trajectories, so ``dualcoh`` is
    retained for reproducing old experiments and must not be described as an
    SNR-identified or Kalman-optimal gate.  New experiments use ``dualwiener``.

    Minimising E|| alpha*g + m*J^T g - (I+J)^T mu ||^2 over BOTH coefficients gives

        [ s_I + n_I/B     b + kappa/B  ] [alpha]   [ s_I + b ]
        [ b + kappa/B     s_J + n_J/B  ] [  m  ] = [ s_J + b ]

    with s_I = ||mu||^2, n_I = E||nu||^2, s_J = ||J^T mu||^2, n_J = E||J^T nu||^2,
    b = <mu, J^T mu>, kappa = E<nu, J^T nu>. The noise powers are divided by B
    because the gate scales a per-sample tensor that is then averaged, so the
    object being denoised is the batch mean, not one draw.

    Every quantity is estimated from the two per-sample gradient matrices, with
    the finite-batch correction applied to each (a squared batch mean carries
    signal + noise/B, so signal = (B*||mean||^2 - E||.||^2)/(B-1); the same
    identity gives the cross term).

    Route I fires first in the backward (it sits one hop from the block output);
    route J fires after traversing the block, so only then are both gradients
    available. The pair solved there is used by the NEXT backward -- a one-step
    lag, matching the SNR gate already in this file. Nothing here syncs to host:
    the coefficients stay 0-dim device tensors and only the periodic log line
    pays a transfer.
    """

    _slot_seq = 0
    _store = {}
    alpha = None
    m = None
    _acc = None          # [sum_a, sum_m, sum_a2, sum_m2] as device tensors
    _n = 0
    _boot_acc = None     # [sum of per-position bootstrap sd] for alpha and m
    _boot_n = 0
    _boot_tick = 0
    _clamp_acc = None    # [a_lo, a_hi, m_lo, m_hi] counts of raw solutions outside [0,1]

    @staticmethod
    def next_slot() -> int:
        _DualGate._slot_seq += 1
        return _DualGate._slot_seq

    @staticmethod
    def forward(ctx, x, which, slot):
        ctx.which = which
        ctx.slot = slot
        return x

    @staticmethod
    def backward(ctx, grad_out):
        # ORDER-AGNOSTIC pairing. autograd pops ready nodes by descending sequence
        # number, and the J probe plus the whole branch chain are created after the
        # I probe, so J can fire first. Whichever arrives second does the solve.
        g = _DualGate.alpha if ctx.which == "I" else _DualGate.m
        held = _DualGate._store.pop(ctx.slot, None)
        if held is None:
            _DualGate._store[ctx.slot] = (ctx.which, grad_out.detach())
            if len(_DualGate._store) > 64:        # unpaired leftovers, do not leak
                _DualGate._store.clear()
        elif held[0] != ctx.which:
            gI = held[1] if held[0] == "I" else grad_out.detach()
            gJ = grad_out.detach() if held[0] == "I" else held[1]
            _DualGate._solve(gI, gJ)
        if g is None:
            return grad_out, None, None
        return grad_out * g.to(grad_out.dtype), None, None

    @staticmethod
    def _pair(a: torch.Tensor, d: torch.Tensor, clamp: bool = True):
        """The 2x2 solution from one pair of per-sample gradient matrices.

        clamp=False returns the RAW solution. The bootstrap must use the raw form:
        once the estimate is pinned to a clamp boundary every resample returns the
        same boundary value, the spread collapses to exactly zero, and the noise
        term is unmeasurable precisely in the regime the runs operate in.
        """
        B = a.shape[0]
        Bf = float(B)
        den = Bf - 1.0
        ma, md_ = a.mean(dim=0), d.mean(dim=0)
        qI = (a * a).sum(dim=1).mean()            # s_I + n_I
        qJ = (d * d).sum(dim=1).mean()            # s_J + n_J
        p = (a * d).sum(dim=1).mean()             # b + kappa
        sI = ((Bf * (ma * ma).sum() - qI) / den).clamp_min(0.0)
        sJ = ((Bf * (md_ * md_).sum() - qJ) / den).clamp_min(0.0)
        bb = (Bf * (ma * md_).sum() - p) / den
        nI = (qI - sI).clamp_min(0.0)
        nJ = (qJ - sJ).clamp_min(0.0)
        kk = p - bb
        A11 = sI + nI / Bf
        A22 = sJ + nJ / Bf
        A12 = bb + kk / Bf
        r1 = sI + bb
        r2 = sJ + bb
        det = A11 * A22 - A12 * A12
        eps = 1e-20
        ok = det.abs() > eps
        al = torch.where(ok, (r1 * A22 - A12 * r2) / (det + eps), sI / (A11 + eps))
        mm = torch.where(ok, (A11 * r2 - A12 * r1) / (det + eps), sJ / (A22 + eps))
        if clamp:
            return al.clamp(0.0, 1.0), mm.clamp(0.0, 1.0)
        return al, mm

    @staticmethod
    def _solve(gI: torch.Tensor, gJ: torch.Tensor) -> None:
        a = gI.reshape(gI.shape[0], -1).float()
        d = gJ.reshape(gJ.shape[0], -1).float()
        B = a.shape[0]
        if not getattr(_DualGate, "_solve_announced", False):
            _DualGate._solve_announced = True
            import sys as _s
            _s.stderr.write("[dual2x2] first solve: gI%s gJ%s B=%d\n"
                            % (tuple(a.shape), tuple(d.shape), B))
            _s.stderr.flush()
        if B < 2 or a.shape != d.shape:
            import sys as _s2
            _s2.stderr.write("[dual2x2] SKIPPED: B=%d shapes %s vs %s\n"
                             % (B, tuple(a.shape), tuple(d.shape)))
            _s2.stderr.flush()
            return
        _al_raw, _mm_raw = _DualGate._pair(a, d, clamp=False)
        _DualGate.alpha = _al_raw.clamp(0.0, 1.0)
        _DualGate.m = _mm_raw.clamp(0.0, 1.0)
        # how often the raw solution falls outside [0,1] and has to be truncated
        _hits = [(_al_raw <= 0).float(), (_al_raw >= 1).float(),
                 (_mm_raw <= 0).float(), (_mm_raw >= 1).float()]
        if _DualGate._clamp_acc is None:
            _DualGate._clamp_acc = [h.clone() for h in _hits]
        else:
            for _i in range(4):
                _DualGate._clamp_acc[_i] = _DualGate._clamp_acc[_i] + _hits[_i]

        # ---- optional: separate ESTIMATION NOISE from real position/state spread --
        # The within-window sd mixes two things: sampling error at a fixed position,
        # and genuine variation of the true coefficients across (block, step, state).
        # Only the second is what routing is supposed to exploit. Bootstrapping the
        # batch rows at ONE position holds the true value fixed, so the spread across
        # resamples is pure estimation noise; subtracting it in quadrature from the
        # total leaves the part that actually varies.
        #   RESGRAD_BOOT=R        resamples per measured solve (0 = off)
        #   RESGRAD_BOOT_EVERY=N  measure every Nth solve (default 20)
        import os as _os0
        try:
            _R = int(float(_os0.environ.get("RESGRAD_BOOT", "0") or 0))
        except ValueError:
            _R = 0
        if _R >= 2:
            try:
                _every = max(int(float(_os0.environ.get("RESGRAD_BOOT_EVERY", "20") or 20)), 1)
            except ValueError:
                _every = 20
            _DualGate._boot_tick += 1
            if _DualGate._boot_tick % _every == 0:
                als, mms = [], []
                for _ in range(_R):
                    idx = torch.randint(0, B, (B,), device=a.device)
                    ai, di = a[idx], d[idx]
                    _al, _mm = _DualGate._pair(ai, di, clamp=False)
                    als.append(_al); mms.append(_mm)
                As = torch.stack(als); Ms = torch.stack(mms)
                sda = As.std(unbiased=False)      # noise sd at THIS position
                sdm = Ms.std(unbiased=False)
                if _DualGate._boot_acc is None:
                    _DualGate._boot_acc = [sda.clone(), sdm.clone()]
                    _DualGate._boot_n = 1
                else:
                    _DualGate._boot_acc[0] = _DualGate._boot_acc[0] + sda
                    _DualGate._boot_acc[1] = _DualGate._boot_acc[1] + sdm
                    _DualGate._boot_n += 1

        # accumulate first and second moments so the log can report the WITHIN-window
        # spread: sd/mean >> 1 means the applied coefficient is dominated by estimation
        # noise rather than by the quantity it is supposed to track.
        a_, m_ = _DualGate.alpha, _DualGate.m
        if _DualGate._acc is None:
            _DualGate._acc = [a_.clone(), m_.clone(), (a_ * a_).clone(), (m_ * m_).clone()]
        else:
            _DualGate._acc[0] = _DualGate._acc[0] + a_
            _DualGate._acc[1] = _DualGate._acc[1] + m_
            _DualGate._acc[2] = _DualGate._acc[2] + a_ * a_
            _DualGate._acc[3] = _DualGate._acc[3] + m_ * m_
        _DualGate._n += 1
        import os as _os
        try:
            win = int(float(_os.environ.get("RESGRAD_LOG_G", "0") or 0))
        except ValueError:
            win = 0
        if win <= 1:
            win = 5000
        if _DualGate._n >= win:
            import sys as _sys
            n = _DualGate._n
            av = float(_DualGate._acc[0].cpu()) / n     # the only syncs, once per window
            mv = float(_DualGate._acc[1].cpu()) / n
            a2 = float(_DualGate._acc[2].cpu()) / n
            m2 = float(_DualGate._acc[3].cpu()) / n
            asd = (max(a2 - av * av, 0.0)) ** 0.5
            msd = (max(m2 - mv * mv, 0.0)) ** 0.5
            line = ("[dual2x2] n=%d  alpha=%.4f+-%.4f (sd/mean=%.2f)  m=%.4f+-%.4f (sd/mean=%.2f)"
                    % (n, av, asd, asd / max(av, 1e-9), mv, msd, msd / max(mv, 1e-9)))
            if _DualGate._boot_acc is not None and _DualGate._boot_n > 0:
                # total_sd^2 = noise_sd^2 + position_sd^2, so the part that genuinely
                # varies across (block, step, state) is what is left after removing
                # the sampling error measured at a fixed position.
                bn = _DualGate._boot_n
                na = float(_DualGate._boot_acc[0].cpu()) / bn
                nm = float(_DualGate._boot_acc[1].cpu()) / bn
                pa = max(asd * asd - na * na, 0.0) ** 0.5
                pm = max(msd * msd - nm * nm, 0.0) ** 0.5
                line += ("\n[dual2x2]   split (boot n=%d, pre-clamp): alpha noise=%.4f pos=%.4f | "
                         "m noise=%.4f pos=%.4f" % (bn, na, pa, nm, pm))
                _DualGate._boot_acc = None
                _DualGate._boot_n = 0
            if _DualGate._clamp_acc is not None:
                cl = [float(x.cpu()) / n * 100.0 for x in _DualGate._clamp_acc]
                line += ("\n[dual2x2]   clamp hits: alpha <=0 %.0f%%  >=1 %.0f%%  |  "
                         "m <=0 %.0f%%  >=1 %.0f%%" % tuple(cl))
                _DualGate._clamp_acc = None
            _sys.stderr.write(line + "\n")
            _sys.stderr.flush()
            _CoherenceGate.ema_g = 0.5 * (av + mv)      # for resgrad_gate_mean
            _DualGate._acc = None
            _DualGate._n = 0


def _bcast(g: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Shape a gate so it broadcasts against `ref`.

    A 0-dim gate broadcasts already. A per-batch-element gate [B] does NOT: the
    branch is [B, d_model] but the recurrent states are [B, d_inner, d_conv] and
    [B, d_inner, d_state], so the trailing dims must be added per target.
    """
    if g.dim() == 0:
        return g
    return g.reshape(-1, *([1] * (ref.dim() - 1)))


def detach_mamba_stack_state(h: MambaStackState) -> MambaStackState:
    return tuple((conv.detach(), ssm.detach()) for conv, ssm in h)


class _CoherenceGate(torch.autograd.Function):
    """Legacy gradient-coherence heuristic (not an SNR estimator).

    Forward is identity; backward scales the incoming gradient by

        c = || E_i g_i ||^2 / E_i || g_i ||^2  in [0, 1],

    where i indexes the batch.  It equals a scalar Wiener ratio only under the
    strong shared-mean/IID-noise assumption; heterogeneous examples violate
    that identification.  It remains available solely for legacy ablations.
    With B=1 it is always fully open."""

    last_c = 1.0
    ema_g = None          # EMA of the APPLIED gate -> reported as resgrad_gate_mean (emergent g)
    _hist = []
    cur_tag = ""          # set by the caller to label which route this gate sits on
    _tag_acc = {}
    _tag_n = 0

    @staticmethod
    def forward(ctx, x):
        ctx.tag = _CoherenceGate.cur_tag
        return x

    @staticmethod
    def backward(ctx, grad_out):
        g = grad_out.reshape(grad_out.shape[0], -1).float()      # [B, D] per-sample grads
        B = g.shape[0]
        num = (g.mean(dim=0) ** 2).sum()                          # ||E_i g_i||^2
        den = (g * g).sum(dim=1).mean() + 1e-12                   # E_i ||g_i||^2
        c = num / den                                            # empirical coherence (biased up ~1/B)
        if B > 1:
            # finite-sample bias correction: pure noise gives E[c]=1/B, not 0, because
            # the mean of B independent noise samples is not exactly zero. c_corr maps
            # noise->0 and signal->1 with no tuned parameter (B is known).
            c = (B * c - 1.0) / (B - 1.0)
        c = c.clamp(0.0, 1.0)

        # --- legacy shared-mean model for the batch-averaged gradient -----------------
        # Only under the shared-mu/IID-noise model rejected above would c estimate
        # S/(S+N), with S=||mu||^2 and N=E||nu||^2. Under that model the averaged
        # gradient carries noise N/B_eff and its corresponding scalar gain would be
        #     m* = S/(S + N/B_eff) = B_eff*c / (B_eff*c + 1 - c).
        # This algebra is retained only to reproduce the legacy ablation; it does not
        # identify SNR for heterogeneous states or trajectories. The bias correction
        # above merely removes finite-B bias within that already restrictive model.
        # RESGRAD_COH_ACCUM: set to grad_accum_steps to account for gradient accumulation
        # (the update averages over local_batch * grad_accum samples). Default 1 = use the
        # micro-batch only, which under-opens (conservative).
        import os as _os
        accum = float(_os.environ.get("RESGRAD_COH_ACCUM", "1") or 1.0)
        b_eff = max(float(B) * accum, 1.0)
        m = (b_eff * c) / (b_eff * c + (1.0 - c) + 1e-12)
        m = m.clamp(0.0, 1.0)

        # ---- bookkeeping, WITHOUT a device sync ------------------------------
        # The gate itself needs none: `m` is a tensor and the multiply below stays
        # on device. Every float(...)/.cpu() here would be a GPU->CPU sync, and the
        # gate fires (routes x layers x rollout steps) times per training step --
        # 512 for the dual gate at K=64, depth 4 -- so syncing per call costs far
        # more than the backward pass. Accumulate on device instead and sync once
        # per window, when a line is actually printed.
        md = m.detach()
        cd = c.detach()
        tag = getattr(ctx, "tag", "") or "_"
        a = _CoherenceGate._tag_acc.get(tag)
        if a is None:
            _CoherenceGate._tag_acc[tag] = [md, cd, 1]
        else:
            a[0] = a[0] + md; a[1] = a[1] + cd; a[2] += 1
        _CoherenceGate._tag_n += 1

        try:
            _win = int(float(_os.environ.get("RESGRAD_LOG_G", "0") or 0))
        except ValueError:
            _win = 0
        if _win <= 1:
            _win = 20000
        if _CoherenceGate._tag_n >= _win:
            import sys as _sys
            parts, tot_m, tot_n = [], 0.0, 0
            for t in sorted(_CoherenceGate._tag_acc):
                sm, sc, n = _CoherenceGate._tag_acc[t]
                gm = float(sm.cpu()) / n          # the only syncs, once per window
                cm = float(sc.cpu()) / n
                parts.append("%s: gate=%.4f raw_c=%.5f n=%d" % (t, gm, cm, n))
                tot_m += gm * n; tot_n += n
            label = "dual gate" if len(parts) > 1 else "coherence g"
            _sys.stderr.write("[%s] B_eff=%.0f | %s\n" % (label, b_eff, " | ".join(parts)))
            _sys.stderr.flush()
            # refresh the float the forward pass reports as resgrad_gate_mean.
            # Stale by at most one window -- it is a diagnostic, not a control.
            if tot_n:
                _CoherenceGate.ema_g = tot_m / tot_n
                _CoherenceGate.last_c = _CoherenceGate.ema_g
            _CoherenceGate._tag_acc = {}
            _CoherenceGate._tag_n = 0
        return grad_out * m.to(grad_out.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class StatefulMambaBlock(nn.Module):
    """One pure-PyTorch Mamba block in step/inference mode.

    Input/output token shape: [B, d_model].  The persistent recurrent state is
    (conv_state, ssm_state).  This follows the official Mamba update equations
    without fused CUDA kernels.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
        norm_type: str = "layer",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        if norm_type == "rms":
            self.norm = RMSNorm(self.d_model)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(self.d_model)
        else:
            raise ValueError(f"Unknown norm_type={norm_type!r}; expected 'layer' or 'rms'.")

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=conv_bias,
            padding=0,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters(dt_init_floor=dt_init_floor)

    def reset_parameters(self, dt_init_floor: float = 1e-4):
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(self.dt_min), math.log(self.dt_max))
        ).clamp(min=float(dt_init_floor))
        with torch.no_grad():
            self.dt_proj.bias.copy_(_inverse_softplus(dt))

    def init_state(self, batch_size: int, device=None, dtype=None) -> MambaLayerState:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def step(
        self,
        token: torch.Tensor,
        state: MambaLayerState,
        residual_grad_gate: float | torch.Tensor = 1.0,
        resgrad_policy: str = "all",
        resgrad_ratio_threshold: float = 0.05,
        forward_branch_scale: float = 1.0,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        route_layer: int = -1,
    ):
        conv_state, ssm_state = state
        import os as _os
        _pol0 = str(resgrad_policy).lower()
        dual_pair = None
        if _pol0 == "dualwiener":
            if dual_wiener is None:
                raise ValueError("resgrad_policy='dualwiener' requires a DualWienerController")
            residual, token_for_branch = dual_wiener.route_pair(
                token, route_horizon, route_layer
            )
            token = self.norm(token_for_branch)
            # The recurrent states are part of the nonlinear route. Gate their
            # temporal inputs so local parameter gradients remain unscaled.
            conv_state = dual_wiener.branch_state_input(
                conv_state, route_horizon, route_layer
            )
            ssm_state = dual_wiener.branch_state_input(
                ssm_state, route_horizon, route_layer
            )
            dual_pair = dual_wiener.current_pair(
                route_horizon, route_layer, token
            )
        elif _pol0 == "dualcoh":
            # LEGACY coherence-derived dual gate. The input tensor is referenced twice -- once as the
            # untouched residual, once through the norm into the branch -- so the
            # two routes are separable here. Both probes share a slot id so the
            # solver can pair them. This does not identify SNR unless the batch
            # satisfies the shared-mean/IID-noise assumption; use dualwiener for
            # the explicit Gaussian residual-noise model.
            _slot = _DualGate.next_slot()
            residual = _DualGate.apply(token, "I", _slot)
            token = self.norm(_DualGate.apply(token, "J", _slot))
        elif _os.environ.get("RESGRAD_MEASURE_DUAL"):
            # measurement only: same split, no change to any gradient
            residual = _PathProbe.apply(token, "I")
            token = self.norm(_PathProbe.apply(token, "J"))
        else:
            residual = token
            token = self.norm(token)

        xz = self.in_proj(token)
        x, z_gate = xz.chunk(2, dim=-1)

        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[:, :, -1] = x
        weight = self.conv1d.weight.squeeze(1)
        x_conv = torch.sum(conv_state * weight.unsqueeze(0), dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt_raw, B_t, C_t = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype, device=x_conv.device)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)
        ssm_state = ssm_state * dA + x_conv.unsqueeze(-1) * dB

        y = torch.einsum("bdn,bn->bd", ssm_state, C_t)
        y = y + self.D.to(dtype=y.dtype, device=y.device).unsqueeze(0) * x_conv
        y = y * self.act(z_gate)
        out = self.out_proj(y)
        branch = self.dropout(out)

        # Residual-gradient routing. Forward value is still
        #     residual + branch
        # but the gradient through the nonlinear branch is optionally removed
        # or scaled. The identity residual path always remains open.
        #
        # Dynamic ratio policy:
        #   At the exact residual addition point, compare the nonlinear branch
        #   magnitude with the identity residual magnitude. If the branch is
        #   small relative to the identity path, long-gradient credit skips the
        #   branch and flows only through the identity edge. This is a hard
        #   scalar decision per block-step, so PyTorch can really drop the
        #   branch backward graph when gate=0.
        policy = str(resgrad_policy).lower()
        branch_norm = branch.detach().float().reshape(branch.shape[0], -1).norm(dim=1).mean()
        residual_norm = residual.detach().float().reshape(residual.shape[0], -1).norm(dim=1).mean()
        branch_residual_ratio = branch_norm / (residual_norm + 1e-8)

        if policy == "dualwiener":
            # Both routes were already gated at their inputs. Do not detach the
            # outputs: that would double-count m and suppress local theta grads.
            effective_gate_for_state = 1.0
            # Keep diagnostics on-device.  This block is called thousands of
            # times per minibatch; converting this scalar to a Python float here
            # forces one CUDA synchronization per layer and rollout step.
            effective_gate_scalar = dual_pair[1].detach()
        elif policy == "dualcoh":
            # alpha and m were already applied at the two route references above;
            # gating `branch` again here would double-count m. The recurrent carry
            # still needs a gate, so hand it to the sentinel path below.
            effective_gate_for_state = None
            effective_gate_scalar = (
                1.0 if _CoherenceGate.ema_g is None else float(_CoherenceGate.ema_g))
        elif policy == "coherence":
            # Legacy cross-start alignment heuristic, computed in the backward
            # pass while keeping the forward value exact. It is not a Wiener/SNR
            # estimate for heterogeneous rollout starts; retain only for ablations.
            branch = _CoherenceGate.apply(branch)
            effective_gate_for_state = None   # sentinel -> coherence-gate the states below
            # Report the EMA of the applied coherence gate (not 1.0) so the emergent open
            # fraction g shows up as resgrad_gate_mean in the normal training logs.
            effective_gate_scalar = (
                1.0 if _CoherenceGate.ema_g is None else float(_CoherenceGate.ema_g)
            )
        else:
            if policy in ("ratio", "act_ratio", "dynamic", "dynamic_ratio", "branch_ratio", "norm_ratio"):
                base_gate = float(residual_grad_gate) if not torch.is_tensor(residual_grad_gate) else float(residual_grad_gate.detach().float().mean().cpu())
                gate_val = 1.0 if float(branch_residual_ratio.detach().cpu()) >= float(resgrad_ratio_threshold) else base_gate
            else:
                gate_val = residual_grad_gate

            # gate = 0: y = residual + stopgrad(branch) in backward
            # gate = 1: ordinary residual block backward
            if torch.is_tensor(gate_val):
                gate_tensor = gate_val.to(device=branch.device, dtype=branch.dtype)
                branch = branch.detach() + _bcast(gate_tensor, branch) * (branch - branch.detach())
                effective_gate_for_state = gate_tensor
                effective_gate_scalar = gate_tensor.detach().float().mean()
            else:
                gate_float = float(gate_val)
                effective_gate_scalar = gate_float
                if gate_float <= 0.0:
                    branch = branch.detach()
                elif gate_float < 1.0:
                    branch = branch.detach() + gate_float * (branch - branch.detach())
                # gate >= 1 keeps ordinary autograd.
                effective_gate_for_state = gate_float
        # forward_branch_scale (default 1.0) scales the branch's FORWARD value
        # only -- used by the inference-time branch-ablation diagnostic to turn
        # the nonlinear residual branch down/off (scale=0 => identity block) and
        # measure how much the model relies on it. 1.0 leaves forward unchanged.
        if forward_branch_scale != 1.0:
            out = residual + float(forward_branch_scale) * branch
        else:
            out = residual + branch

        # Apply the same residual-gradient gate to the recurrent state carried
        # to later rollout steps.  This is essential for memory: if the output
        # branch is detached but conv_state/ssm_state still keep their graph,
        # future steps can still backpropagate through the whole nonlinear
        # Mamba state-update chain.
        if effective_gate_for_state is None:
            # coherence policy: gate the recurrent-state gradient by its own coherence too.
            conv_state_for_next = _CoherenceGate.apply(conv_state)
            ssm_state_for_next = _CoherenceGate.apply(ssm_state)
        elif torch.is_tensor(effective_gate_for_state):
            gate_tensor = effective_gate_for_state.to(device=ssm_state.device, dtype=ssm_state.dtype)
            conv_state_for_next = conv_state.detach() + _bcast(gate_tensor, conv_state) * (conv_state - conv_state.detach())
            ssm_state_for_next = ssm_state.detach() + _bcast(gate_tensor, ssm_state) * (ssm_state - ssm_state.detach())
        else:
            gate_float = float(effective_gate_for_state)
            if gate_float <= 0.0:
                conv_state_for_next = conv_state.detach()
                ssm_state_for_next = ssm_state.detach()
            elif gate_float < 1.0:
                conv_state_for_next = conv_state.detach() + gate_float * (conv_state - conv_state.detach())
                ssm_state_for_next = ssm_state.detach() + gate_float * (ssm_state - ssm_state.detach())
            else:
                conv_state_for_next = conv_state
                ssm_state_for_next = ssm_state

        gate_metric = (
            effective_gate_scalar.detach()
            if torch.is_tensor(effective_gate_scalar)
            else branch.new_tensor(float(effective_gate_scalar))
        )
        aux = {
            "dt_mean": dt.mean(),
            "dt_min": dt.min(),
            "dt_max": dt.max(),
            "ssm_state_norm": ssm_state.reshape(ssm_state.shape[0], -1).norm(dim=1).mean(),
            "conv_state_norm": conv_state.reshape(conv_state.shape[0], -1).norm(dim=1).mean(),
            "resgrad_gate": gate_metric,
            "resgrad_branch_norm": branch_norm.detach(),
            "resgrad_residual_norm": residual_norm.detach(),
            "resgrad_branch_residual_ratio": branch_residual_ratio.detach(),
        }
        if dual_pair is not None:
            aux["dual_wiener_alpha"] = dual_pair[0].detach()
            aux["dual_wiener_m"] = dual_pair[1].detach()
        return out, (conv_state_for_next, ssm_state_for_next), aux


class OfficialStateMambaARModel(nn.Module, _StimPoolMixin):
    """Official-state Mamba AR model for vector or 2D-field states."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_state_mamba = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: int | str = "auto",
        mamba_dt_min: float = 1e-3,
        mamba_dt_max: float = 1e-1,
        mamba_norm_type: str = "layer",
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        resgrad_block_gate: float = 1.0,
        resgrad_ratio_threshold: float = 0.05,
        resgrad_keep_every: int = 8,
        resgrad_keep_tail: int = 0,
        resgrad_outer: bool = False,
        untie_groups: int = 1,
        predict_sigma: bool = False,
        dual_wiener_ema: float = 0.95,
        dual_wiener_residual_ema: float = 0.99,
        dual_wiener_warmup_batches: int = 8,
        dual_wiener_probe_every: int = 4,
        dual_wiener_min_probes: int = 1,
        dual_wiener_noise_model: str = "diagonal_gaussian",
        dual_wiener_max_horizon: int = 1024,
        global_horizon_wiener: bool = False,
        global_wiener_ridge: float = 1e-8,
        global_wiener_anchor: float = 0.0,
        global_wiener_local_fidelity: float = 0.0,
        global_wiener_solver_iters: int = 256,
        global_wiener_sketch_dim: int = 8192,
        global_wiener_sketch_seed: int = 1729,
        global_wiener_noise_draws: int = 4,
        global_wiener_batch_conditioned: bool = True,
        global_wiener_superbatch_groups: int = 1,
        global_wiener_static_gain: float = -1.0,
        global_wiener_static_mode: str = "delayed_tied",
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        # Heteroscedastic spread head for proper-scoring (Gaussian CRPS) training.
        self.predict_sigma = bool(predict_sigma)
        self.task_type = "field2d" if self.state_shape is not None and len(self.state_shape) == 3 else "vector"
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        self.resgrad_block_gate = float(resgrad_block_gate)
        self.resgrad_ratio_threshold = float(resgrad_ratio_threshold)
        self.resgrad_keep_every = max(1, int(resgrad_keep_every))
        self.resgrad_keep_tail = max(0, int(resgrad_keep_tail))
        # New noise-identified dual route estimator.  It is instantiated only
        # for the new policy, so old checkpoints keep their exact state_dict.
        self.dual_wiener = None
        if self.resgrad_routing and self.resgrad_policy == "dualwiener":
            if bool(resgrad_outer):
                raise ValueError(
                    "dualwiener is an internal per-block policy; do not combine "
                    "it with --resgrad_outer"
                )
            self.dual_wiener = DualWienerController(
                state_dim=self.state_dim,
                depth=self.depth,
                max_horizon=int(dual_wiener_max_horizon),
                ema=float(dual_wiener_ema),
                residual_ema=float(dual_wiener_residual_ema),
                warmup_batches=int(dual_wiener_warmup_batches),
                probe_every=int(dual_wiener_probe_every),
                min_probes=int(dual_wiener_min_probes),
                noise_model=str(dual_wiener_noise_model),
            )
        self.global_horizon_wiener = None
        if bool(global_horizon_wiener):
            if self.dual_wiener is not None:
                raise ValueError(
                    "global-horizon Wiener and internal dualwiener routing are "
                    "alternative backward operators; enable only one"
                )
            self.global_horizon_wiener = GlobalHorizonWienerController(
                state_dim=self.state_dim,
                max_horizon=int(dual_wiener_max_horizon),
                ema=float(dual_wiener_ema),
                residual_ema=float(dual_wiener_residual_ema),
                warmup_batches=int(dual_wiener_warmup_batches),
                probe_every=int(dual_wiener_probe_every),
                min_probes=int(dual_wiener_min_probes),
                noise_model=str(dual_wiener_noise_model),
                ridge=float(global_wiener_ridge),
                anchor=float(global_wiener_anchor),
                local_fidelity=float(global_wiener_local_fidelity),
                solver_iterations=int(global_wiener_solver_iters),
                sketch_dim=int(global_wiener_sketch_dim),
                sketch_seed=int(global_wiener_sketch_seed),
                noise_draws=int(global_wiener_noise_draws),
                batch_conditioned=bool(global_wiener_batch_conditioned),
                superbatch_groups=int(global_wiener_superbatch_groups),
                static_gain=float(global_wiener_static_gain),
                static_mode=str(global_wiener_static_mode),
            )
        # Temporal-residual routing: instead of gating the INTERNAL residual
        # branches (whose branch<->time map is muddy on a 4-block Mamba step),
        # keep every internal block fully open and gate the backward of the
        # OUTER per-step residual delta in  pred = x_t + Delta = x_t + out(z).
        # This is the theory's object exactly: the per-step Jacobian is
        # J = d(x+Delta)/dx = I + dDelta/dx, so scaling Delta's backward by m in
        # {0,1} realizes the mechanism's I + m*J_F one-to-one, with no
        # branch<->step assumption. Forward value x_t+Delta is unchanged. The
        # keep/cut policy reuses resgrad_policy (head/tail/periodic/fixed) plus
        # an outer dynamic-ratio ||Delta||/||x_t|| analog of the block ratio.
        self.resgrad_outer = bool(resgrad_outer)
        # Inference-time forward branch scaling (1.0 = no change). Set by the
        # branch-ablation diagnostic to y = x + scale*F(x) per block.
        self.resgrad_forward_branch_scale = 1.0
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1
        # SNR-gate state: the previous rollout step's ||Delta||/||sigma|| ratio.
        # sigma is produced only after the step's blocks run, so the SNR gate uses a
        # one-step lag; reset to None at the rollout start (horizon_index==0).
        self._resgrad_snr_prev = None
        # Running (EMA) reference SNR for the RELATIVE snrk gate: the gate opens around
        # the data's own typical SNR rather than around an absolute 1, so low-information
        # data does not starve the backward gradient (see step()). Not reset per rollout.
        self._resgrad_snr_ref = None
        # snrk cold-start guards (env-driven so no argparse/builder plumbing):
        #   RESGRAD_SNR_WARMUP = N: keep the gate FULLY OPEN for the first N training
        #     rollouts, so the model (and its sigma head) can learn before the SNR gate
        #     engages -- you cannot estimate SNR from a model that predicts nothing.
        #   RESGRAD_SNR_FLOOR = f: never let the gate fall below f, so the backward
        #     gradient (hence drive-learning, which shares the recurrent block) is never
        #     fully starved. This is what fMRI needs: history~0 there, but the gate must
        #     not collapse and kill drive-learning too.
        import os as _os
        self.resgrad_snr_warmup_steps = int(_os.environ.get("RESGRAD_SNR_WARMUP", "0"))
        self.resgrad_snr_floor = float(_os.environ.get("RESGRAD_SNR_FLOOR", "0.0"))
        self._resgrad_fwd_count = 0
        # Weight-sharing knob for the temporal-untying experiment. untie_groups=1
        # (default) is the usual fully-shared rollout: one block stack applied at
        # every step. untie_groups=G>1 gives G INDEPENDENT block-stack copies; a
        # rollout step k (of total_horizon K) uses group min(k*G//K, G-1), so the
        # first K/G steps use group 0, the next K/G steps group 1, etc. G=K fully
        # unties the rollout. This tests whether routing's near-losslessness comes
        # from parameter reuse: as G rises, each block's gradient loses the local
        # terms from steps assigned to other groups, so its delayed terms are no
        # longer redundant and routing should cost more.
        self.untie_groups = max(1, int(untie_groups))

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        def _make_stack():
            return nn.ModuleList([
                StatefulMambaBlock(
                    d_model=self.hidden_dim,
                    d_state=int(mamba_d_state),
                    d_conv=int(mamba_d_conv),
                    expand=int(mamba_expand),
                    dt_rank=mamba_dt_rank,
                    dt_min=float(mamba_dt_min),
                    dt_max=float(mamba_dt_max),
                    dropout=float(dropout),
                    norm_type=mamba_norm_type,
                )
                for _ in range(max(1, self.depth))
            ])

        # Backward compatibility: with untie_groups=1 the module tree is byte-for-byte
        # the original single `self.blocks` stack, so every existing checkpoint still
        # loads. Only untie_groups>1 (fresh untying experiments) builds `block_groups`
        # of G independent, architecture-identical copies; the recurrent state flows
        # continuously across a group switch. Exactly one of the two is non-None.
        if self.untie_groups == 1:
            self.blocks = _make_stack()
            self.block_groups = None
        else:
            self.blocks = None
            self.block_groups = nn.ModuleList([_make_stack() for _ in range(self.untie_groups)])
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        # Optional per-coordinate spread head off the same final features z, for
        # Gaussian-CRPS training. Built only when enabled, so MSE-trained
        # checkpoints (no sigma head) still load with strict=True.
        self.sigma_out = None
        if self.predict_sigma:
            self.sigma_out = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.state_dim),
            )

    def _active_stack(self, horizon_index: Optional[int] = None):
        """The depth-L block stack to use at this rollout step.

        untie_groups=1 -> always self.blocks. Otherwise pick group
        min(k*G//K, G-1) from horizon_index k and total_horizon K; if either is
        unknown (e.g. burn-in / one-step forward) fall back to group 0."""
        if self.block_groups is None:
            return self.blocks
        G = len(self.block_groups)
        k = int(horizon_index) if horizon_index is not None else int(getattr(self, "resgrad_current_horizon", -1))
        K = int(getattr(self, "resgrad_current_total_horizon", -1))
        if k < 0 or K <= 0:
            return self.block_groups[0]
        return self.block_groups[min((k * G) // K, G - 1)]

    def init_state(self, batch_size: int, device=None, dtype=None) -> MambaStackState:
        stack = self.blocks if self.block_groups is None else self.block_groups[0]
        return tuple(block.init_state(batch_size, device=device, dtype=dtype) for block in stack)

    def detach_state(self, h: MambaStackState) -> MambaStackState:
        return detach_mamba_stack_state(h)

    def _flatten_state(self, x_t: torch.Tensor) -> torch.Tensor:
        return x_t.reshape(x_t.shape[0], -1)

    def _reshape_pred(self, pred_flat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return pred_flat.reshape(ref.shape)

    def _pool_stim_point(self, stim_t: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        if not self.has_external_input:
            return torch.zeros(B, 0, device=device, dtype=dtype)
        if stim_t is None:
            return torch.zeros(B, self.input_dim, device=device, dtype=dtype)
        if stim_t.dim() == 1:
            stim_t = stim_t.unsqueeze(0)
        pooled = self._pool_stim(stim_t.unsqueeze(1), B, 1, device, dtype)
        return pooled[:, 0]

    def make_token(self, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]) -> torch.Tensor:
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_flat, stim], dim=-1) if self.has_external_input else x_flat
        return self.in_proj(inp)

    def _resgrad_gate_for_step(self, horizon_index: Optional[int] = None) -> float:
        """Return the nonlinear-branch gradient gate for this rollout step.

        This affects only residual branches inside the Mamba stack.  The
        identity shortcut always propagates gradients.  Policies are intended
        for long/BPTT training only; evaluation/no-grad forward values are
        unchanged.
        """
        # Even in eval/no-grad, compute the configured gate for diagnostics.
        # Forward values are unchanged and eval has no backward graph, but logging
        # gate=1 unconditionally makes dynamic routing look inactive.
        if not self.resgrad_routing:
            return 1.0
        pol = str(self.resgrad_policy).lower()
        k = int(horizon_index) if horizon_index is not None else int(getattr(self, "resgrad_current_horizon", -1))
        K = int(getattr(self, "resgrad_current_total_horizon", -1))
        base_gate = float(self.resgrad_block_gate)
        if pol in ("all", "full", "normal"):
            return 1.0
        if pol in ("none", "identity", "id"):
            return 0.0
        if pol in ("fixed", "scalar"):
            return base_gate
        if pol in ("periodic", "every"):
            if k < 0:
                return base_gate
            return 1.0 if (k % self.resgrad_keep_every == 0) else base_gate
        if pol in ("tail", "last"):
            if k < 0 or K <= 0:
                return base_gate
            return 1.0 if k >= max(0, K - self.resgrad_keep_tail) else base_gate
        if pol in ("head", "first"):
            if k < 0:
                return base_gate
            return 1.0 if k < self.resgrad_keep_tail else base_gate
        return base_gate

    def set_resgrad_context(self, horizon_index: Optional[int] = None, total_horizon: Optional[int] = None):
        self.resgrad_current_horizon = -1 if horizon_index is None else int(horizon_index)
        self.resgrad_current_total_horizon = -1 if total_horizon is None else int(total_horizon)

    # ---- dual-Wiener training hooks -------------------------------------
    # The trainer invokes these around the ordinary loss backward.  Probe
    # gradients are obtained with autograd.grad, so they never accumulate in
    # model parameters.
    def dual_wiener_begin_batch(self) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.begin_batch()

    def dual_wiener_probe_terms(self, prediction, target, horizon_index):
        if self.dual_wiener is None or not self.training:
            return None, None
        return self.dual_wiener.probe_terms(prediction, target, horizon_index)

    def dual_wiener_set_probe_losses(self, total, noise) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.set_probe_losses(total, noise)

    def dual_wiener_calibrate(self) -> bool:
        return bool(self.dual_wiener is not None and self.dual_wiener.calibrate())

    def dual_wiener_end_batch(self) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.end_batch()

    def dual_wiener_diagnostics(self, horizon: int):
        if self.dual_wiener is None:
            return {}
        return self.dual_wiener.diagnostics(horizon)

    def dual_wiener_export_state(self, horizon: int):
        if self.dual_wiener is None:
            return None
        return self.dual_wiener.export_state(horizon)

    # ---- global-horizon Wiener training hooks ---------------------------
    def global_wiener_begin_batch(self) -> None:
        if self.global_horizon_wiener is not None:
            self.global_horizon_wiener.begin_batch()

    def global_wiener_observe_and_sample(self, prediction, target, horizon_index):
        if self.global_horizon_wiener is None or not self.training:
            return None
        return self.global_horizon_wiener.observe_and_sample(
            horizon_index, prediction, target
        )

    def global_wiener_add_probe_terms(self, horizon_index, total, noise) -> None:
        if self.global_horizon_wiener is not None:
            self.global_horizon_wiener.add_probe_terms(
                horizon_index, total, noise
            )

    def global_wiener_weight_loss(self, loss, horizon_index):
        if self.global_horizon_wiener is None:
            return loss
        return self.global_horizon_wiener.weight_loss(loss, horizon_index)

    def global_wiener_calibrate(self, parameters) -> bool:
        return bool(
            self.global_horizon_wiener is not None
            and self.global_horizon_wiener.calibrate(parameters)
        )

    def global_wiener_end_batch(self) -> None:
        if self.global_horizon_wiener is not None:
            self.global_horizon_wiener.end_batch()

    def global_wiener_diagnostics(self, horizon: int):
        if self.global_horizon_wiener is None:
            return {}
        return self.global_horizon_wiener.diagnostics(horizon)

    def global_wiener_export_state(self, horizon: int):
        if self.global_horizon_wiener is None:
            return None
        return self.global_horizon_wiener.export_state(horizon)

    def step(self, h: Optional[MambaStackState], x_t: torch.Tensor, stim_t: Optional[torch.Tensor] = None, return_aux: bool = False, horizon_index: Optional[int] = None, total_horizon: Optional[int] = None, ratio_collector: Optional[list] = None):
        B = x_t.shape[0]
        if h is None:
            h = self.init_state(B, x_t.device, x_t.dtype)
        # REACH gate. Applied to x_t before anything else, so it covers BOTH
        # temporal routes out of this step: the token fed to the blocks and the
        # AR skip x_flat used in the residual add below. Forward value unchanged.
        alpha = _skip_alpha(horizon_index)
        if alpha < 1.0:
            x_t = _apply_alpha(x_t, alpha)
        # "cohall": the DERIVED single-coefficient gate. Minimising the same
        # bias-variance objective over one scalar multiplying the whole step,
        # beta(I+J), gives the Wiener gain of the SUMMED path -- and grad_out at
        # this point already IS the total incoming gradient, so the statistic is
        # unchanged from the branch-only gate; only the attachment moves. The
        # consequence is that the accumulated gain becomes prod beta instead of
        # being floored at 1 by the bare identity term.
        if str(self.resgrad_policy).lower() == "cohall":
            x_t = _CoherenceGate.apply(x_t)
        token = self.make_token(x_t, stim_t)
        if horizon_index is not None or total_horizon is not None:
            self.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)

        next_states: List[MambaLayerState] = []
        dt_means, dt_mins, dt_maxs = [], [], []
        ssm_norms, conv_norms = [], []
        resgrad_gates, resgrad_ratios, resgrad_branch_norms, resgrad_residual_norms = [], [], [], []
        dual_wiener_alphas, dual_wiener_ms = [], []
        z = token
        gate = self._resgrad_gate_for_step(horizon_index)
        pol = str(self.resgrad_policy).lower()
        global _POL_ANNOUNCED
        if not _POL_ANNOUNCED:
            _POL_ANNOUNCED = True
            import sys as _s
            _s.stderr.write("[resgrad] routing=%r policy=%r outer=%r block_gate=%r\n"
                            % (getattr(self, "resgrad_routing", None), self.resgrad_policy,
                               getattr(self, "resgrad_outer", None),
                               getattr(self, "resgrad_block_gate", None)))
            _s.stderr.flush()
        snr_mode = pol in ("snr", "snrk")   # both gate by ||Delta||/||sigma||
        snrk_mode = pol == "snrk"           # soft Kalman-gain gate (else hard threshold)
        if snr_mode:
            # SNR gate on the INTERNAL branches, from the PREVIOUS step's
            # signal-to-noise ||Delta||/||sigma|| (sigma follows the blocks, hence the
            # one-step lag; reset at the rollout start). Blocks apply the scalar ("all").
            if horizon_index is not None and int(horizon_index) <= 0:
                self._resgrad_snr_prev = None
                if snrk_mode and self.training:
                    self._resgrad_fwd_count += 1
            prev = self._resgrad_snr_prev
            # WARMUP: keep the gate fully open for the first RESGRAD_SNR_WARMUP training
            # rollouts, so the model + sigma head can learn before the SNR gate engages.
            in_warmup = (snrk_mode and self.training
                         and self._resgrad_fwd_count <= self.resgrad_snr_warmup_steps)
            if prev is None or in_warmup:
                block_gate = 1.0
            elif snrk_mode:
                # RELATIVE soft Kalman gain: compare this step's SNR to the data's own
                # running SNR level (ref, an EMA over training), not to absolute 1. The
                # absolute gate SNR^2/(1+SNR^2) demands SNR~1 to open; on low-information
                # data (SNR<1 everywhere -- which is WHY the model is hard to train) it
                # collapses to ~0 and starves the backward gradient in a cold-start death
                # spiral (no gradient -> no learning -> low SNR -> gate stays shut).
                # Gating relative to ref keeps ~half the credit flowing (the steps whose
                # SNR beats the typical level) so g emerges from the SHAPE of the
                # SNR-vs-horizon curve, not from an absolute threshold the data never meets.
                ref = self._resgrad_snr_ref
                r2 = (ref * ref) if (ref is not None and ref > 1e-6) else (prev * prev)
                block_gate = (prev * prev) / (r2 + prev * prev)
                # FLOOR: never fully close -- keeps drive-learning (which shares the
                # recurrent block) from being starved when history-SNR collapses (fMRI).
                if self.resgrad_snr_floor > 0.0:
                    block_gate = max(float(self.resgrad_snr_floor), block_gate)
            else:
                block_gate = 1.0 if prev >= float(self.resgrad_ratio_threshold) else float(self.resgrad_block_gate)
            block_policy = "all"
        elif pol == "cohall":
            # the single coefficient was already applied to x_t at the top of this
            # step, which scales the WHOLE map beta(I+J). The blocks must then do
            # nothing extra -- gating the branch again would re-introduce the
            # asymmetry between the two routes that this policy exists to remove.
            block_gate = 1.0
            block_policy = "all"
        elif pol == "oracle":
            # EXPERIMENT 2 -- the ceiling. The mask comes from the model-free
            # per-(start, horizon) predictability map e_k(t), thresholded WITHIN
            # each k, so it is pure state selection at every horizon and never a
            # decaying k-profile (that would be truncation). The gate is a
            # per-batch-element tensor; blocks apply it verbatim ("all").
            from .oracle_gate import OracleGate
            _orc = OracleGate.maybe()
            if _orc is None:
                raise SystemExit(
                    "--resgrad_policy oracle requires RESGRAD_ORACLE=<ek_map.npz> "
                    "(build it with scripts/compute_ek_map.py --oracle-fracs ...)")
            if horizon_index is not None and int(horizon_index) <= 0:
                _orc.begin_rollout(x_t)
            block_gate = _orc.gate(int(horizon_index or 0), B, x_t.device, x_t.dtype)
            block_policy = "all"
        elif self.resgrad_outer:
            # Temporal-residual (outer) mode: internal blocks fully open; the gate is
            # applied to the outer Delta after self.out below.
            block_gate = 1.0
            block_policy = "all"
        else:
            block_gate = gate
            block_policy = self.resgrad_policy
        # tell the measurement probe which rollout depth this forward belongs to
        _PathProbe.cur_k = int(horizon_index) if horizon_index is not None else -1
        active_blocks = self._active_stack(horizon_index)
        for layer_index, (block, layer_state) in enumerate(zip(active_blocks, h)):
            z, next_state, aux_l = block.step(
                z,
                layer_state,
                residual_grad_gate=block_gate,
                resgrad_policy=block_policy,
                resgrad_ratio_threshold=self.resgrad_ratio_threshold,
                forward_branch_scale=self.resgrad_forward_branch_scale,
                dual_wiener=self.dual_wiener,
                route_horizon=int(horizon_index) if horizon_index is not None else -1,
                route_layer=layer_index,
            )
            next_states.append(next_state)
            dt_means.append(aux_l["dt_mean"])
            dt_mins.append(aux_l["dt_min"])
            dt_maxs.append(aux_l["dt_max"])
            ssm_norms.append(aux_l["ssm_state_norm"])
            conv_norms.append(aux_l["conv_state_norm"])
            if "resgrad_gate" in aux_l:
                resgrad_gates.append(aux_l["resgrad_gate"])
            if "resgrad_branch_residual_ratio" in aux_l:
                resgrad_ratios.append(aux_l["resgrad_branch_residual_ratio"])
                # In snr mode the calibrator must see ONLY the outer SNR ||Delta||/||sigma||
                # (appended below), not the per-block magnitude ratios, or the threshold
                # gets set on the wrong distribution and the gate opens far too few steps.
                if ratio_collector is not None and not snr_mode:
                    ratio_collector.append(float(aux_l["resgrad_branch_residual_ratio"].detach().cpu()))
            if "resgrad_branch_norm" in aux_l:
                resgrad_branch_norms.append(aux_l["resgrad_branch_norm"])
            if "resgrad_residual_norm" in aux_l:
                resgrad_residual_norms.append(aux_l["resgrad_residual_norm"])
            if "dual_wiener_alpha" in aux_l:
                dual_wiener_alphas.append(aux_l["dual_wiener_alpha"])
            if "dual_wiener_m" in aux_l:
                dual_wiener_ms.append(aux_l["dual_wiener_m"])
        # The recurrent carry is the other half of the temporal path, so the reach
        # gate must cut it too -- otherwise the Mamba state keeps the graph alive
        # across the boundary and the "segment detach" is only partial.
        if alpha < 1.0:
            next_states = [(_apply_alpha(c, alpha), _apply_alpha(s, alpha))
                           for c, s in next_states]
        if pol == "cohall":
            # the recurrent carry is the other temporal route out of this step,
            # so it takes the same single coefficient as the AR skip
            next_states = [(_CoherenceGate.apply(c), _CoherenceGate.apply(s))
                           for c, s in next_states]
        h_next = tuple(next_states)

        delta_or_frame = self.out(z)
        x_flat = self._flatten_state(x_t)
        # Temporal-residual routing: gate the BACKWARD of the outer per-step
        # residual Delta = delta_or_frame. Forward value x_t+Delta is unchanged
        # (straight-through: d.detach() + g*(d - d.detach()) == d in value).
        if self.residual and self.resgrad_outer and self.resgrad_routing:
            pol = str(self.resgrad_policy).lower()
            if pol in ("ratio", "act_ratio", "dynamic", "dynamic_ratio", "branch_ratio", "norm_ratio"):
                dn = delta_or_frame.detach().float().reshape(B, -1).norm(dim=1).mean()
                xn = x_flat.detach().float().reshape(B, -1).norm(dim=1).mean()
                ratio = float((dn / (xn + 1e-8)).cpu())
                # Feed the outer ratio to the open-fraction calibrator (no-grad
                # rollout), so --resgrad_target_open_frac tunes the outer threshold.
                if ratio_collector is not None:
                    ratio_collector.append(ratio)
                g_outer = 1.0 if ratio >= float(self.resgrad_ratio_threshold) else float(self.resgrad_block_gate)
            else:
                g_outer = float(gate)  # positional (head/tail/periodic) / fixed / none
            if g_outer < 1.0:
                delta_or_frame = delta_or_frame.detach() + g_outer * (delta_or_frame - delta_or_frame.detach())
            gate = g_outer
        pred_flat = x_flat + delta_or_frame if self.residual else delta_or_frame
        pred = self._reshape_pred(pred_flat, x_t)

        if snr_mode:
            # This step's signal-to-noise = ||Delta|| / ||sigma||: the reducible
            # update (Delta, the well-fit mean) over the estimated irreducible noise
            # (sigma). Stored for the NEXT step's gate and fed to the open-fraction
            # calibrator. Unlike the magnitude ratio ||Delta||/||x||, this cuts steps
            # whose update is large only because their noise (sigma) is large.
            if self.sigma_out is None:
                raise ValueError("resgrad_policy='snr' needs the sigma head; build with predict_sigma=True (--mamba_crps)")
            sigma_flat = torch.nn.functional.softplus(self.sigma_out(z)) + 1e-3
            dn = delta_or_frame.detach().float().reshape(B, -1).norm(dim=1)
            sn = sigma_flat.detach().float().reshape(B, -1).norm(dim=1)
            self._resgrad_snr_prev = float((dn / (sn + 1e-8)).mean().cpu())
            if snrk_mode and self.training:
                # EMA of the typical SNR -- the reference the relative gate opens around.
                cur = self._resgrad_snr_prev
                self._resgrad_snr_ref = cur if self._resgrad_snr_ref is None \
                    else 0.99 * self._resgrad_snr_ref + 0.01 * cur
            if ratio_collector is not None:
                ratio_collector.append(self._resgrad_snr_prev)

        if return_aux:
            sigma = None
            if self.predict_sigma and self.sigma_out is not None:
                sigma_flat = torch.nn.functional.softplus(self.sigma_out(z)) + 1e-3
                sigma = self._reshape_pred(sigma_flat, x_t)
            aux = {
                "sigma": sigma,
                # Frozen representation used by post-hoc innovation probes.  It
                # is exposed only on the explicit return_aux path and detached
                # so a nuisance head cannot update, or retain the graph of, the
                # forecasting model.
                "innovation_features": z.detach(),
                "h_next": h_next,
                "hidden_norm": torch.stack(ssm_norms).mean() if ssm_norms else pred_flat.new_tensor(0.0),
                "mamba_ssm_state_norm": torch.stack(ssm_norms).mean() if ssm_norms else pred_flat.new_tensor(0.0),
                "mamba_conv_state_norm": torch.stack(conv_norms).mean() if conv_norms else pred_flat.new_tensor(0.0),
                "mamba_dt_mean": torch.stack(dt_means).mean() if dt_means else pred_flat.new_tensor(0.0),
                "mamba_dt_min": torch.stack(dt_mins).min() if dt_mins else pred_flat.new_tensor(0.0),
                "mamba_dt_max": torch.stack(dt_maxs).max() if dt_maxs else pred_flat.new_tensor(0.0),
                "alpha_mean": torch.stack(dt_means).mean() if dt_means else pred_flat.new_tensor(0.0),
                "alpha_min": torch.stack(dt_mins).min() if dt_mins else pred_flat.new_tensor(0.0),
                "alpha_max": torch.stack(dt_maxs).max() if dt_maxs else pred_flat.new_tensor(0.0),
                "resgrad_gate": torch.stack(resgrad_gates).mean() if resgrad_gates else pred_flat.new_tensor(float(gate)),
                # Per-layer gates for this step (diagnostics only; the mean above is
                # unchanged). Shape [depth]; empty tensor if no per-layer gate logged.
                "resgrad_gate_per_layer": torch.stack(resgrad_gates) if resgrad_gates else pred_flat.new_zeros(0),
                "resgrad_routing": pred_flat.new_tensor(1.0 if self.resgrad_routing else 0.0),
                "resgrad_branch_residual_ratio": torch.stack(resgrad_ratios).mean() if resgrad_ratios else pred_flat.new_tensor(0.0),
                "resgrad_branch_norm": torch.stack(resgrad_branch_norms).mean() if resgrad_branch_norms else pred_flat.new_tensor(0.0),
                "resgrad_residual_norm": torch.stack(resgrad_residual_norms).mean() if resgrad_residual_norms else pred_flat.new_tensor(0.0),
                "dual_wiener_alpha": torch.stack(dual_wiener_alphas).mean() if dual_wiener_alphas else pred_flat.new_tensor(1.0),
                "dual_wiener_m": torch.stack(dual_wiener_ms).mean() if dual_wiener_ms else pred_flat.new_tensor(1.0),
            }
            return pred, h_next, aux
        return pred, h_next

    def burn_in(self, x_seq: torch.Tensor, stim_seq: Optional[torch.Tensor] = None, h0: Optional[MambaStackState] = None, detach: bool = False):
        B, T = x_seq.shape[:2]
        h = h0 if h0 is not None else self.init_state(B, x_seq.device, x_seq.dtype)
        ctx = torch.no_grad() if detach else torch.enable_grad()
        with ctx:
            for j in range(T):
                stim_j = stim_seq[:, j] if stim_seq is not None else None
                _, h = self.step(h, x_seq[:, j], stim_j, return_aux=False)
        return self.detach_state(h) if detach else h

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, W = history.shape[:2]
        h = self.init_state(B, history.device, history.dtype)
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype) if self.has_external_input else None
        pred = None
        for j in range(W):
            stim_j = stim[:, j] if stim is not None else None
            pred, h = self.step(h, history[:, j], stim_j, return_aux=False)
        if pred is None:
            raise ValueError("history must contain at least one frame")
        return pred

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred
