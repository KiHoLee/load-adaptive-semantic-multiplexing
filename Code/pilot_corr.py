# =========================================================
# pilot_corr.py — pilot (2026-11-03): channel-aware placement of the dynamic masks under fading
# that is correlated across the slots of a block. Not part of the submitted manuscript.
#
# Channel: every user u has its own amplitude |h_u[s]| per slot s, an AR(1) complex Gaussian
# process over the N = 24 slots with correlation rho between adjacent slots and CN(0,1)
# marginals, phase pre-compensated per slot. The scheduler (receiver) knows |h_u[s]| of every
# active user, announces the placement in the grant, and scales each received slot by the
# MMSE factor g / (g^2 + sigma^2) of its owner before the gather. Every scheme gets this receiver.
#
# Placement modes (one user per slot, rank j keeps the quota N_j = ceil((N - j)/U) of the fixed rule):
#   fixed     slot s -> rank s mod U, segment floor(s/U)            (the manuscript rule)
#   block     slot s -> rank floor(s/(N/U)), contiguous parts          (classic static partition)
#   csi       Hungarian slot sets maximizing sum log(1 + snr g^2), segments in slot order
#   csisort   the same slot sets, leading segment on the strongest slot of the user
#   learn     Hungarian on learned per-(load, segment) weights w * log(1 + snr g^2); gradients
#             reach w through a straight-through Sinkhorn relaxation of the assignment
#   net       a mask network takes the channel state of every active user and slot (gain, its rank
#             and offset within the user, the gains of the other users on the slot, SNR, load) and a
#             per-(load, segment) embedding, scores every (slot, user, segment) triple, and the
#             Hungarian assignment of the scores is the placement (straight-through Sinkhorn in training)
#   netpw     net plus per-slot transmit power weights and receive weights from the same channel state
#             (each user still spends d_s/U, distributed over its slots by the network)
#   netq      prop only: the mask network also sets HOW MANY slots each user gets in this frame (1 to
#             twice its fair share), which only a nested code allows; a user then sends its leading
#             n_j segments, strongest slot first, at unit average power over its n_j occupied slots
#   netqpw    netq plus the per-slot transmit and receive weights of netpw
#
# Optional near-far channel (--nearfar X): every user and frame gets an average-SNR offset uniform in
# [-X, X] dB, in training and evaluation, and the channel state seen by the network includes it.
#
# Schemes: prop (nested code + dynamic masks + SNR profile + shared decoder, one model for
# U = 1..6) and ods<k> (Static ODS-k, k exclusive parts of N/k slots, trained and evaluated at U = k).
#
# Usage: python pilot_corr.py --scheme prop --mode csisort --seed 1 [--epochs 150]
#        python pilot_corr.py --selftest
# Output: ../data/pilot_corr/<scheme>_<mode>_s<seed>.json
# =========================================================
import argparse, json, math, os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from rev_lib import get_device, set_seed, embedding_cache, sample_frames, semantic_loss, TRAIN_SNRS, EVAL_SNRS, DATA

OUT = os.path.join(DATA, "pilot_corr")


# ---------------------------------------------------------
# Channel and placement
# ---------------------------------------------------------
def slot_gains(B, U, S, rho, device):
    """|h_u[s]|, AR(1) over slots with CN(0,1) marginals, shape (B,U,S)."""
    w = torch.complex(torch.randn(B, U, S, device=device), torch.randn(B, U, S, device=device)) / math.sqrt(2)
    if rho >= 1.0:
        return w[..., :1].abs().expand(B, U, S).contiguous()
    h = [w[..., 0]]
    a = math.sqrt(1.0 - rho ** 2)
    for s in range(1, S):
        h.append(rho * h[-1] + a * w[..., s])
    return torch.stack(h, -1).abs()


def quotas(U, S):
    return [math.ceil((S - j) / U) for j in range(U)]


def pairs(U, S, quota=None):
    """pair q = (rank j, segment r) in rank-major order; returns jq, rq (M,) and qidx (U,S) with -1 padding.
    quota: segments per rank, default the quotas of the fixed rule (M = S)."""
    jq, rq, qidx = [], [], -torch.ones(U, S, dtype=torch.long)
    for j, n in enumerate(quota or quotas(U, S)):
        for r in range(n):
            qidx[j, r] = len(jq)
            jq.append(j); rq.append(r)
    return torch.tensor(jq), torch.tensor(rq), qidx


def place(G, U, S, mode, snr_db, w=None, k=None):
    """Hard placement as a one-hot (B,S,M) matrix P[b, slot, pair]. G (B,U,S) by rank. With k (Static
    ODS-k at a load U <= k) every rank has S/k segments and the slots of absent parts stay idle."""
    B = G.shape[0]
    quota = [S // k] * U if k else None
    jq, rq, qidx = pairs(U, S, quota)
    M = len(jq)
    P = torch.zeros(B, S, M)
    if mode in ("fixed", "block"):
        s = torch.arange(S)
        kk = k or U
        if mode == "fixed":
            j, r = s % kk, s // kk
        else:
            assert S % kk == 0
            j, r = s // (S // kk), s % (S // kk)
        keep = j < U
        P[:, s[keep], qidx[j[keep], r[keep]]] = 1.0
        return P
    Gc = G.detach().cpu().double()
    util = torch.log1p(10 ** (snr_db / 10.0) * Gc ** 2)                             # (B,U,S)
    Gq = util[:, jq, :].transpose(1, 2)                                             # (B,S slots,M pairs)
    if mode == "learn":
        cost = -(Gq * w.detach().cpu().double()[rq].view(1, 1, M))
    else:
        cost = -Gq
    cost, Gc, jqn, qn = cost.numpy(), Gc.numpy(), jq.numpy(), qidx.numpy()
    for b in range(B):
        rows, cols = linear_sum_assignment(cost[b])
        if mode in ("csi", "csisort"):
            # equal weights: keep each user's slot set, then order its segments
            owner = jqn[cols]
            for j in range(U):
                sl = rows[owner == j]
                sl = np.sort(sl) if mode == "csi" else sl[np.argsort(-Gc[b, j, sl], kind="stable")]
                P[b, torch.as_tensor(sl), torch.as_tensor(qn[j, :len(sl)])] = 1.0
        else:
            P[b, rows, cols] = 1.0
    return P


def sinkhorn(logits, iters=20):
    """doubly stochastic relaxation of an assignment, log domain, (B,S,M) with M <= S: the M - S
    missing pairs are dummy columns of logit 0 (idle slots), dropped from the result."""
    B, S, M = logits.shape
    la = torch.cat([logits, torch.zeros(B, S, S - M, device=logits.device)], 2) if M < S else logits
    for _ in range(iters):
        la = la - torch.logsumexp(la, dim=2, keepdim=True)
        la = la - torch.logsumexp(la, dim=1, keepdim=True)
    return la.exp()[:, :, :M]


def through_channel(Xq, Gq_t, P, snr_db):
    """Xq (B,S pairs,seg) transmitted symbols per pair, Gq_t (B,S slots,S pairs) owner gains,
    P (B,S,S) placement. Returns the MMSE-scaled received symbols per pair (B,S,seg)."""
    sigma2 = 1.0 / (10 ** (snr_db / 10.0))
    A = P * Gq_t                                                   # gain of the pair on its slot
    Y = torch.einsum("bsq,bqd->bsd", A, Xq)
    Y = Y + math.sqrt(sigma2) * torch.randn_like(Y)
    go = A.sum(-1, keepdim=True)                                   # owner gain per slot
    Yc = Y * go / (go ** 2 + sigma2)
    return torch.einsum("bsq,bsd->bqd", P, Yc)


# ---------------------------------------------------------
# Channel-state mask network (modes net, netpw)
# ---------------------------------------------------------
NF, NE, NH = 10, 8, 64


def csi_features(Gr, snr_db, U):
    """(B,U,S) gains by rank -> (B,U,S,NF) features of user j on slot s."""
    B, _, S = Gr.shape
    u = torch.log1p(10 ** (snr_db / 10.0) * Gr ** 2) / 5.0
    mu, mx = u.mean(-1, keepdim=True), u.amax(-1, keepdim=True)
    rk = torch.argsort(torch.argsort(-u, dim=-1), dim=-1).float() / (S - 1)
    if U > 1:
        om = (u.sum(1, keepdim=True) - u) / (U - 1)
        eye = torch.eye(U, device=u.device, dtype=torch.bool).view(1, U, U, 1)
        ox = u.unsqueeze(1).expand(B, U, U, S).masked_fill(eye, -1e9).amax(2)
    else:
        om = torch.zeros_like(u); ox = torch.zeros_like(u)
    t = torch.full_like(u, (min(snr_db, 25.0) - 15.0) / 15.0)
    lf = torch.full_like(u, U / 6.0)
    return torch.stack([u, mu.expand_as(u), mx.expand_as(u), u - mu, rk, om, ox, u - ox, t, lf], -1)


def csi_net_init(m, n_loads, S, pw):
    def lin(name, o, i, bias=0.0):
        w = torch.empty(o, i); nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        setattr(m, name + "_w", nn.Parameter(w)); setattr(m, name + "_b", nn.Parameter(torch.full((o,), bias)))
    m.mk_seg = nn.Parameter(0.1 * torch.randn(n_loads, S, NE))
    lin("mk1", NH, NF + NE); lin("mk2", 1, NH)
    if pw:
        sp1 = math.log(math.e - 1.0)                       # softplus(sp1) = 1 at start
        lin("pw1", 32, NF); lin("pw2", 1, 32, sp1); lin("rw1", 32, NF); lin("rw2", 1, 32, sp1)
        for n in ("pw2_w", "rw2_w"):
            getattr(m, n).data.mul_(0.01)


def mlp(p, x, a, b):
    return F.linear(F.relu(F.linear(x, p[a + "_w"], p[a + "_b"])), p[b + "_w"], p[b + "_b"]).squeeze(-1)


def csi_net_apply(p, Gr, snr_db, U, li, jq, rq, S, pw, tau):
    """placement P (B,S,S) and per-pair transmit / receive weights (B,S) or None."""
    f = csi_features(Gr, snr_db, U)                                         # (B,U,S,NF)
    fq = f[:, jq].transpose(1, 2)                                           # (B,S slots,M pairs,NF)
    M = len(jq)
    er = p["mk_seg"][li][rq].view(1, 1, M, NE).expand(fq.shape[0], S, M, NE)
    score = mlp(p, torch.cat([fq, er], -1), "mk1", "mk2")                   # (B,S,M)
    sc = score.detach().cpu().double().numpy()
    P = torch.zeros(sc.shape)                                               # built on the CPU, moved once
    for b in range(sc.shape[0]):
        rows, cols = linear_sum_assignment(-sc[b])
        P[b, torch.as_tensor(rows), torch.as_tensor(cols)] = 1.0
    P = P.to(score.device)
    if torch.is_grad_enabled():
        lg = score / (score.std(dim=(1, 2), keepdim=True) + 1e-6) / tau
        Ps = sinkhorn(lg)
        P = P + Ps - Ps.detach()                                            # straight-through
    if not pw:
        return P, None, None
    a = F.softplus(mlp(p, f, "pw1", "pw2"))[:, jq].transpose(1, 2)          # (B,S slots,S pairs)
    r = F.softplus(mlp(p, f, "rw1", "rw2"))[:, jq].transpose(1, 2)
    return P, (P * a).sum(1), (P * r).sum(1)


def per_user_energy(Xq, jq, U, energy):
    """scale the pairs of every user to the total energy `energy`."""
    en = torch.zeros(Xq.shape[0], U, device=Xq.device).index_add(1, jq, Xq.pow(2).sum(-1))
    return Xq * torch.sqrt(energy / (en[:, jq] + 1e-8)).unsqueeze(-1)


# ---------------------------------------------------------
# Transceivers
# ---------------------------------------------------------
class PropCorr(nn.Module):
    """Proposed (one model) of the manuscript (shared_dyns) with channel-aware placement."""
    def __init__(self, U_max, d_b, d_s, S=24, mode="fixed", tau=0.1):
        super().__init__()
        self.U_max, self.d_b, self.d_s, self.S, self.seg, self.mode, self.tau = U_max, d_b, d_s, S, d_s // S, mode, tau
        self.tx_w = nn.Parameter(torch.empty(d_s, d_b)); self.tx_b = nn.Parameter(torch.zeros(d_s))
        self.tx_ln_w = nn.Parameter(torch.ones(d_s)); self.tx_ln_b = nn.Parameter(torch.zeros(d_s))
        self.tx_gain = nn.Parameter(torch.ones(U_max, d_s)); self.rx_gain = nn.Parameter(torch.ones(U_max, d_s))
        self.tx_slope = nn.Parameter(torch.zeros(U_max, d_s)); self.rx_slope = nn.Parameter(torch.zeros(U_max, d_s))
        self.rx_w = nn.Parameter(torch.empty(1, d_b, d_s)); self.rx_b = nn.Parameter(torch.zeros(1, d_b))
        if mode == "learn":   # per-(load, segment) importance, slightly decreasing at start
            self.imp = nn.Parameter(torch.log(torch.expm1(1.0 - torch.arange(S).float() / (2 * S))).repeat(U_max, 1))
        if mode in ("net", "netpw"):
            csi_net_init(self, U_max, S, mode == "netpw")
        nn.init.kaiming_uniform_(self.rx_w.data[0], a=math.sqrt(5)); nn.init.kaiming_uniform_(self.tx_w, a=math.sqrt(5))
        nn.init.uniform_(self.tx_b, -1 / math.sqrt(d_b), 1 / math.sqrt(d_b)); nn.init.uniform_(self.rx_b, -1 / math.sqrt(d_s), 1 / math.sqrt(d_s))

    def params(self):
        return {k: v for k, v in self.named_parameters()}

    def forward(self, b, idx, snr_db, G, p=None):
        p = self.params() if p is None else p
        B, U, _ = b.shape
        S, seg, dev = self.S, self.seg, b.device
        order = torch.argsort(idx, dim=1)                                        # position of rank j
        br = b[torch.arange(B, device=dev).unsqueeze(1), order]
        Gr = G[torch.arange(B, device=dev).unsqueeze(1), order]
        jq, rq, qidx = [t.to(dev) for t in pairs(U, S)]
        t = (min(snr_db, 25.0) - 15.0) / 15.0
        e = F.layer_norm(F.linear(br, p["tx_w"], p["tx_b"]), (self.d_s,), p["tx_ln_w"], p["tx_ln_b"])
        e = (e * p["tx_gain"][U - 1] * torch.exp(p["tx_slope"][U - 1] * t)).reshape(B, U, S, seg)
        valid = (qidx >= 0).float().to(dev)                                      # (U,S) segment r < N_j
        e = e * valid.view(1, U, S, 1)
        Xq = e[:, jq, rq]                                                        # (B,S pairs,seg)
        Gq_t = Gr[:, jq, :].transpose(1, 2)                                      # (B,S slots,S pairs)
        amp = ramp = None
        if self.mode in ("net", "netpw"):
            P, amp, ramp = csi_net_apply(p, Gr, snr_db, U, U - 1, jq, rq, S, self.mode == "netpw", self.tau)
        else:
            w = F.softplus(p["imp"][U - 1]) if self.mode == "learn" else None
            P = place(Gr, U, S, self.mode, snr_db, w).to(dev)
            if self.mode == "learn" and torch.is_grad_enabled():
                util = torch.log1p(10 ** (snr_db / 10.0) * Gq_t ** 2)
                lg = util * w[rq].view(1, 1, S)
                lg = lg / (lg.std(dim=(1, 2), keepdim=True) + 1e-6) / self.tau
                Ps = sinkhorn(lg)
                P = P + Ps - Ps.detach()                                         # straight-through
        if amp is not None:
            Xq = Xq * amp.unsqueeze(-1)
        Xq = per_user_energy(Xq, jq, U, self.d_s / U)                            # energy d_s/U per user
        Zq = through_channel(Xq, Gq_t, P, snr_db)
        if ramp is not None:
            Zq = Zq * ramp.unsqueeze(-1)
        z = torch.cat([Zq, torch.zeros(B, 1, seg, device=dev)], 1)[:, qidx.clamp(min=-1) % (S + 1)]
        z = (z * valid.view(1, U, S, 1)).reshape(B, U, self.d_s)
        vm = valid.repeat_interleave(seg, dim=-1).view(1, U, self.d_s)
        n = vm.sum(-1, keepdim=True)
        mu = (z * vm).sum(-1, keepdim=True) / n
        var = (((z - mu) * vm) ** 2).sum(-1, keepdim=True) / n
        z = (z - mu) / torch.sqrt(var + 1e-5) * vm
        z = z * p["rx_gain"][U - 1] * torch.exp(p["rx_slope"][U - 1] * t) * vm
        bh_r = F.linear(z, p["rx_w"][0], p["rx_b"][0])
        bh = torch.empty_like(bh_r)
        bh[torch.arange(B, device=dev).unsqueeze(1), order] = bh_r
        return bh


class PropQ(PropCorr):
    """Proposed (one model) with channel-adaptive slot counts (modes netq, netqpw). Pairs (j, r) run
    over r < R_U = min(N, 2 ceil(N/U)); the Hungarian assignment of the network scores picks one pair
    per slot, which fixes n_j, the number of slots of user j (segment 0 of every user carries a large
    bonus, so n_j >= 1). User j then sends segments 0..n_j-1 on its slots, strongest slot first."""
    def __init__(self, U_max, d_b, d_s, S=24, mode="netq", tau=0.1):
        super().__init__(U_max, d_b, d_s, S=S, mode="net" if mode == "netq" else "netpw", tau=tau)
        self.mode = mode

    def forward(self, b, idx, snr_db, G, p=None):
        p = self.params() if p is None else p
        B, U, _ = b.shape
        S, seg, dev = self.S, self.seg, b.device
        pw = self.mode == "netqpw"
        ar = torch.arange(B, device=dev).unsqueeze(1)
        order = torch.argsort(idx, dim=1)
        br, Gr = b[ar, order], G[ar, order]
        R = min(S, 2 * math.ceil(S / U))
        jq = torch.arange(U, device=dev).repeat_interleave(R); rq = torch.arange(R, device=dev).repeat(U)
        M = U * R
        t = (min(snr_db, 25.0) - 15.0) / 15.0
        e = F.layer_norm(F.linear(br, p["tx_w"], p["tx_b"]), (self.d_s,), p["tx_ln_w"], p["tx_ln_b"])
        e = (e * p["tx_gain"][U - 1] * torch.exp(p["tx_slope"][U - 1] * t)).reshape(B, U, S, seg)
        f = csi_features(Gr, snr_db, U)                                       # (B,U,S,NF)
        fq = f[:, jq].transpose(1, 2)                                         # (B,S,M,NF)
        er = p["mk_seg"][U - 1][rq].view(1, 1, M, NE).expand(B, S, M, NE)
        raw = mlp(p, torch.cat([fq, er], -1), "mk1", "mk2")
        bonus = (rq == 0).float().view(1, 1, M)                               # every user sends segment 0
        # hard placement: Hungarian over slots x pairs, then relabel each user's segments by slot strength
        sc = (raw + 1e3 * bonus).detach().cpu().double().numpy()
        Gc = Gr.detach().cpu().numpy()
        Ph = torch.zeros(B, S, M)
        nseg = torch.zeros(B, U, dtype=torch.long)
        jqn = jq.cpu().numpy()
        for bb in range(B):
            rows, cols = linear_sum_assignment(-sc[bb])
            owner = jqn[cols]
            for j in range(U):
                sl = rows[owner == j]
                sl = sl[np.argsort(-Gc[bb, j, sl], kind="stable")]
                Ph[bb, torch.as_tensor(sl), j * R + torch.arange(len(sl))] = 1.0
                nseg[bb, j] = len(sl)
        P = Ph.to(dev)
        if torch.is_grad_enabled():
            # square relaxation with M - S dummy slots of score 0 (a pair may stay unsent)
            lg = raw / (raw.std(dim=(1, 2), keepdim=True) + 1e-6) / self.tau + 1e2 * bonus
            lg = torch.cat([lg, torch.zeros(B, M - S, M, device=dev)], 1)
            Ps = sinkhorn(lg)[:, :S]
            P = P + Ps - Ps.detach()
        sent = Ph.sum(1).to(dev)                                              # (B,M) pair sent
        Xq = e[:, jq, rq] * sent.unsqueeze(-1)
        Gq_t = Gr[:, jq, :].transpose(1, 2)                                   # (B,S,M)
        if pw:
            a = F.softplus(mlp(p, f, "pw1", "pw2"))[:, jq].transpose(1, 2)
            Xq = Xq * (P * a).sum(1).unsqueeze(-1)
        n_j = nseg.to(dev).float()
        en = torch.zeros(B, U, device=dev).index_add(1, jq, Xq.pow(2).sum(-1))
        Xq = Xq * torch.sqrt(n_j[:, jq] * seg / (en[:, jq] + 1e-8)).unsqueeze(-1)   # unit power per occupied use
        Zq = through_channel(Xq, Gq_t, P, snr_db)
        if pw:
            r = F.softplus(mlp(p, f, "rw1", "rw2"))[:, jq].transpose(1, 2)
            Zq = Zq * (P * r).sum(1).unsqueeze(-1)
        z = torch.zeros(B, U, S, seg, device=dev)
        z[:, :, :R] = Zq.reshape(B, U, R, seg)
        valid = (torch.arange(S, device=dev).view(1, 1, S) < n_j.unsqueeze(-1)).float()   # (B,U,S)
        z = (z * valid.unsqueeze(-1)).reshape(B, U, self.d_s)
        vm = valid.repeat_interleave(seg, dim=-1)
        n = vm.sum(-1, keepdim=True)
        mu = (z * vm).sum(-1, keepdim=True) / n
        var = (((z - mu) * vm) ** 2).sum(-1, keepdim=True) / n
        z = (z - mu) / torch.sqrt(var + 1e-5) * vm
        z = z * p["rx_gain"][U - 1] * torch.exp(p["rx_slope"][U - 1] * t) * vm
        bh_r = F.linear(z, p["rx_w"][0], p["rx_b"][0])
        bh = torch.empty_like(bh_r)
        bh[ar, order] = bh_r
        self.last_nseg = nseg
        return bh


class OdsCorr(nn.Module):
    """Static ODS-k at its design load U = k: k exclusive parts of N/k slots, learned projection,
    LN, and linear decoder (OrthoDS of the manuscript), with the placement modes above."""
    def __init__(self, k, d_b, d_s, S=24, mode="block", tau=0.1):
        super().__init__()
        assert S % k == 0
        self.k, self.d_b, self.d_s, self.S, self.seg, self.mode, self.tau = k, d_b, d_s, S, d_s // S, mode, tau
        if mode in ("net", "netpw"):
            csi_net_init(self, 1, S, mode == "netpw")
        self.d_u = d_s // k
        d_u = self.d_u
        self.tx_w = nn.Parameter(torch.empty(d_u, d_b)); self.tx_b = nn.Parameter(torch.zeros(d_u))
        self.tx_ln_w = nn.Parameter(torch.ones(d_u)); self.tx_ln_b = nn.Parameter(torch.zeros(d_u))
        self.rx_ln_w = nn.Parameter(torch.ones(d_u)); self.rx_ln_b = nn.Parameter(torch.zeros(d_u))
        self.rx_w = nn.Parameter(torch.empty(d_b, d_u)); self.rx_b = nn.Parameter(torch.zeros(d_b))
        nn.init.kaiming_uniform_(self.tx_w, a=math.sqrt(5)); nn.init.kaiming_uniform_(self.rx_w, a=math.sqrt(5))
        nn.init.uniform_(self.tx_b, -1 / math.sqrt(d_b), 1 / math.sqrt(d_b)); nn.init.uniform_(self.rx_b, -1 / math.sqrt(d_u), 1 / math.sqrt(d_u))

    def params(self):
        return {k: v for k, v in self.named_parameters()}

    def forward(self, b, idx, snr_db, G, p=None):
        p = self.params() if p is None else p
        B, U, _ = b.shape
        assert U <= self.k
        S, seg, dev = self.S, self.seg, b.device
        order = torch.argsort(idx, dim=1)
        br = b[torch.arange(B, device=dev).unsqueeze(1), order]
        Gr = G[torch.arange(B, device=dev).unsqueeze(1), order]
        jq, rq, qidx = [t.to(dev) for t in pairs(U, S, [S // self.k] * U)]
        x = F.layer_norm(F.linear(br, p["tx_w"], p["tx_b"]), (self.d_u,), p["tx_ln_w"], p["tx_ln_b"])
        x = x.reshape(B, U, S // self.k, seg)
        Xq = x[:, jq, rq]
        Gq_t = Gr[:, jq, :].transpose(1, 2)
        amp = ramp = None
        if self.mode in ("net", "netpw"):
            P, amp, ramp = csi_net_apply(p, Gr, snr_db, U, 0, jq, rq, S, self.mode == "netpw", self.tau)
        else:
            P = place(Gr, U, S, self.mode, snr_db, k=self.k).to(dev)
        if amp is not None:
            Xq = Xq * amp.unsqueeze(-1)
        Xq = per_user_energy(Xq, jq, U, float(self.d_u))                         # energy d_s/k per user
        Zq = through_channel(Xq, Gq_t, P, snr_db)
        if ramp is not None:
            Zq = Zq * ramp.unsqueeze(-1)
        z = Zq[:, qidx[:, :S // self.k].reshape(-1)].reshape(B, U, self.d_u)
        z = F.layer_norm(z, (self.d_u,), p["rx_ln_w"], p["rx_ln_b"])
        bh_r = F.linear(z, p["rx_w"], p["rx_b"])
        bh = torch.empty_like(bh_r)
        bh[torch.arange(B, device=dev).unsqueeze(1), order] = bh_r
        return bh


# ---------------------------------------------------------
# Training, adaptation, evaluation (first-order SNR-aware MAML as rev_lib)
# ---------------------------------------------------------
NEARFAR = 0.0


def frames(E, B, U, U_max, rho, device, gen, agnostic):
    b, idx, _ = sample_frames(E, B, U, U_max, device, gen, agnostic)
    G = slot_gains(B, U, 24, rho, device)
    if NEARFAR > 0:
        G = G * 10 ** ((2 * torch.rand(B, U, 1, device=device) - 1) * NEARFAR / 20.0)
    return b, idx, G


def inner(model, p, b, idx, snr, G, alpha, lam):
    loss = semantic_loss(b, model(b, idx, snr, G, p=p), lam)
    keys = list(p.keys())
    grads = torch.autograd.grad(loss, [p[k] for k in keys], allow_unused=True)
    return {k: (p[k] - alpha * g.detach() if g is not None else p[k]) for k, g in zip(keys, grads)}


def train(model, cache, cfg, device, log=print):
    E = cache["train"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["beta"])
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        tot = 0.0
        for _ in range(cfg["steps"]):
            U = int(torch.randint(1, cfg["U_max"] + 1, (1,), generator=gen)) if cfg["agnostic"] else cfg["U"]
            tasks = random.sample(list(TRAIN_SNRS), cfg["meta_batch"])
            base = model.params()
            meta = 0.0
            for snr in tasks:
                b, idx, G = frames(E, cfg["batch"], U, cfg["U_max"], cfg["rho"], device, gen, cfg["agnostic"])
                q = inner(model, dict(base), b, idx, snr, G, cfg["alpha"], cfg["lam"])
                b, idx, G = frames(E, cfg["batch"], U, cfg["U_max"], cfg["rho"], device, gen, cfg["agnostic"])
                meta = meta + semantic_loss(b, model(b, idx, snr, G, p=q), cfg["lam"])
            meta = meta / cfg["meta_batch"]
            opt.zero_grad(set_to_none=True)
            meta.backward()
            opt.step()
            tot += meta.item()
        if ep == 1 or ep % 10 == 0 or ep == cfg["epochs"]:
            log(f"  [{cfg['name']} ep {ep:03d}] loss={tot / cfg['steps']:.4f} t={time.time() - t0:.0f}s")
    return time.time() - t0


def adapt(model, cache, cfg, device, snr, U, rho):
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 + snr)
    p = dict(model.params())
    with torch.enable_grad():
        b, idx, G = frames(cache["train"], 256, U, cfg["U_max"], rho, device, gen, cfg["agnostic"])
        p = inner(model, p, b, idx, snr, G, cfg["alpha"], cfg["lam"])
    return {k: v.detach() for k, v in p.items()}


@torch.no_grad()
def evaluate(model, cache, cfg, device, U, rho, trials=2000, bs=250, eval_seed=123):
    out = {}
    for snr in EVAL_SNRS:
        p = adapt(model, cache, cfg, device, snr, U, cfg["rho"])     # adaptation at the training rho
        set_seed(eval_seed + snr)
        gen = torch.Generator().manual_seed(eval_seed + snr)
        cos = []
        for _ in range(trials // bs):
            b, idx, G = frames(cache["test"], bs, U, cfg["U_max"], rho, device, gen, cfg["agnostic"])
            cos.append(F.cosine_similarity(model(b, idx, snr, G, p=p), b, dim=-1).cpu())
        out[snr] = torch.cat(cos, 0).mean().item()
    return out


def selftest():
    """placement is a permutation, gather inverts it, quotas sum to N, csisort puts segment 0 on the best slot."""
    torch.manual_seed(0)
    S = 24
    for U in range(1, 7):
        G = slot_gains(4, U, S, 0.7, "cpu")
        for mode in ["fixed", "csi", "csisort", "learn"] + (["block"] if S % U == 0 else []):
            w = torch.ones(S) if mode == "learn" else None
            P = place(G, U, S, mode, 10.0, w)
            assert torch.all(P.sum(1) == 1) and torch.all(P.sum(2) == 1), (U, mode)
            Xq = torch.randn(4, S, 128)
            Z = through_channel(Xq, torch.ones(4, S, S), P, 300.0)
            assert (Z - Xq).abs().max() < 1e-3, (U, mode)
            if mode == "csisort":
                jq, rq, qidx = pairs(U, S)
                for b in range(4):
                    for j in range(U):
                        slots = [int(P[b, :, qidx[j, r]].argmax()) for r in range(int((qidx[j] >= 0).sum()))]
                        g = [float(G[b, j, x]) for x in slots]
                        assert g == sorted(g, reverse=True), (U, j)
    for mode in ("net", "netpw"):
        m = PropCorr(6, 16, 96, S=24, mode=mode)
        for U in (1, 3, 6):
            b = F.normalize(torch.randn(4, U, 16), dim=-1); idx = torch.argsort(torch.rand(4, 6), 1)[:, :U]
            G = slot_gains(4, U, 24, 0.7, "cpu")
            loss = semantic_loss(b, m(b, idx, 10.0, G))
            gr = torch.autograd.grad(loss, [m.mk1_w] + ([m.pw1_w] if mode == "netpw" else []))
            assert all(float(g.abs().sum()) > 0 for g in gr), (mode, U)
        mo = OdsCorr(4, 16, 96, S=24, mode=mode)
        b = F.normalize(torch.randn(4, 4, 16), dim=-1); idx = torch.arange(4).repeat(4, 1)
        loss = semantic_loss(b, mo(b, idx, 10.0, slot_gains(4, 4, 24, 0.7, "cpu")))
        assert float(torch.autograd.grad(loss, mo.mk1_w)[0].abs().sum()) > 0, mode
    for mode in ("block", "fixed", "csisort", "netpw"):
        mo = OdsCorr(6, 16, 96, S=24, mode=mode)
        for U in (1, 3, 6):
            b = F.normalize(torch.randn(4, U, 16), dim=-1); idx = torch.argsort(torch.rand(4, 6), 1)[:, :U]
            G = slot_gains(4, U, 24, 0.7, "cpu")
            if mode != "netpw":
                P = place(G, U, 24, mode, 10.0, k=6)
                assert P.shape == (4, 24, 4 * U) and torch.all(P.sum(1) == 1) and torch.all(P.sum(2) <= 1), (mode, U)
            semantic_loss(b, mo(b, idx, 10.0, G))
    for mode in ("netq", "netqpw"):
        m = PropQ(6, 16, 96, S=24, mode=mode)
        for U in (1, 2, 5, 6):
            b = F.normalize(torch.randn(4, U, 16), dim=-1); idx = torch.argsort(torch.rand(4, 6), 1)[:, :U]
            loss = semantic_loss(b, m(b, idx, 10.0, slot_gains(4, U, 24, 0.7, "cpu")))
            ns = m.last_nseg
            assert torch.all(ns.sum(1) == 24) and torch.all(ns >= 1) and torch.all(ns <= min(24, 2 * math.ceil(24 / U))), (mode, U, ns)
            assert float(torch.autograd.grad(loss, m.mk1_w)[0].abs().sum()) > 0, (mode, U)
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="prop")          # prop | ods1 .. ods6
    ap.add_argument("--mode", default="fixed", choices=["fixed", "block", "csi", "csisort", "learn", "net", "netpw", "netq", "netqpw"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--eval-rho", type=float, nargs="+", default=[0.3, 0.7, 0.95, 1.0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--ods-all-loads", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    global NEARFAR
    NEARFAR = a.nearfar
    torch.set_num_threads(2)
    dev = get_device()
    set_seed(a.seed)
    cache = embedding_cache("ag_news", "bert-base-uncased", dev)
    d_b, K = 768, 4
    if a.scheme == "prop":
        model = (PropQ if a.mode in ("netq", "netqpw") else PropCorr)(6, d_b, d_b * K, mode=a.mode).to(dev)
        cfg = dict(U=6, U_max=6, agnostic=True)
        loads = [1, 2, 3, 4, 5, 6]
    else:
        k = int(a.scheme[3:])
        model = OdsCorr(k, d_b, d_b * K, mode=a.mode).to(dev)
        if a.ods_all_loads:      # one Static ODS-k model trained and evaluated at every load U <= k
            cfg = dict(U=k, U_max=k, agnostic=True)
            loads = list(range(1, k + 1))
        else:
            cfg = dict(U=k, U_max=k, agnostic=False)
            loads = [k]
    name = f"{a.scheme}_{a.mode}{a.tag}"
    cfg.update(name=name, seed=a.seed, rho=a.rho, nearfar=a.nearfar, epochs=a.epochs, steps=a.steps, batch=a.batch,
               meta_batch=4, alpha=5e-4, beta=1e-3, lam=0.5)
    t = train(model, cache, cfg, dev)
    res = {str(r): {str(U): evaluate(model, cache, cfg, dev, U, r, trials=a.trials) for U in loads} for r in a.eval_rho}
    os.makedirs(OUT, exist_ok=True)
    rec = {"cfg": cfg, "train_time_s": t, "eval_rho": a.eval_rho, "cos": res}
    if a.mode == "learn":
        rec["imp"] = F.softplus(model.imp.detach()).cpu().tolist()
    json.dump(rec, open(os.path.join(OUT, f"{name}_s{a.seed}.json"), "w"), indent=1)
    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, os.path.join(OUT, f"{name}_s{a.seed}.pt"))
    print("done", name, json.dumps({r: {U: round(v[10], 4) for U, v in d.items()} for r, d in res.items()}))


if __name__ == "__main__":
    main()
