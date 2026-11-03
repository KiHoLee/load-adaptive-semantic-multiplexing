# =========================================================
# todma_redim.py — ToDMA reference with the slot length re-dimensioned at the same block
# (2026-11-03). Slot-batched re-implementation of cl_experiments.py::todma_eval: same codebook
# seed, frame and fading draws, per-slot OMP with exclusion, genie token association, and BERT
# re-encoding of the detected tokens. T slots of L uses with T*L = d_s = 3072; a user sends its
# first T tokens, one per slot at the energy L/U, so it spends at most T*L/U = d_s/U.
#
# Usage: python todma_redim.py --frames 1000 U,T,L [U,T,L ...]   ->  ../data/extra/todma_redim.json
# =========================================================
import argparse, json, math, os, time, torch, torch.nn.functional as F
from cl_experiments import load_agnews_labeled, Extractor, SplitCache, EVAL_SNRS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "extra", "todma_redim.json")


@torch.no_grad()
def ev(bert, cache, dev, U, T, L, n_frames, seed=7, snrs=EVAL_SNRS):
    tok = bert.tokenizer
    g = torch.Generator().manual_seed(seed)
    C = F.normalize(torch.randn(tok.vocab_size, L, generator=g), p=2, dim=1).to(dev)
    amp = math.sqrt(L / U)
    tok_ids = [tok(t, add_special_tokens=False)["input_ids"][:T] for t in cache.test_texts]
    res = {}
    for snr in snrs:
        sigma = math.sqrt(1.0 / (10 ** (snr / 10.0)))
        rng = torch.Generator().manual_seed(seed + snr)
        cos, n_err, n_tok = [], 0, 0
        for _ in range(n_frames):
            idx = torch.randint(0, len(cache.test), (U,), generator=rng)
            seqs = [tok_ids[i] for i in idx.tolist()]
            hr = torch.randn(U, generator=rng); hi = torch.randn(U, generator=rng)
            h = torch.sqrt(hr ** 2 + hi ** 2) / math.sqrt(2.0)
            Tmax = max(max(len(s) for s in seqs), 1)
            ids = torch.zeros(Tmax, U, dtype=torch.long); act = torch.zeros(Tmax, U)
            for u, s in enumerate(seqs):
                if s:
                    ids[:len(s), u] = torch.tensor(s); act[:len(s), u] = 1
            ids, act, hd = ids.to(dev), act.to(dev), (h * amp).to(dev)
            Y = (C[ids] * (hd[None, :, None] * act[:, :, None])).sum(1) + sigma * torch.randn(Tmax, L, device=dev)
            R, sup, Q = Y.clone(), [], []
            for _k in range(U):
                corr = (R @ C.T).abs()
                if sup:
                    S = torch.stack(sup, 1)
                    corr.scatter_(1, S, torch.full_like(S, -1, dtype=corr.dtype))
                j = corr.argmax(1); sup.append(j)
                q = C[j]
                for qq in Q:
                    q = q - (q * qq).sum(1, keepdim=True) * qq
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9); Q.append(q)
                R = R - (R * q).sum(1, keepdim=True) * q
            S = torch.stack(sup, 1).cpu()
            nact = act.sum(1).long().cpu()
            det = [[] for _ in range(U)]
            for t in range(Tmax):
                sset = set(S[t, :int(nact[t])].tolist())
                for u in range(U):
                    if t < len(seqs[u]):
                        n_tok += 1
                        if seqs[u][t] in sset:
                            det[u].append(seqs[u][t])
                        else:
                            n_err += 1
            texts = [tok.decode(d) if d else "[UNK]" for d in det]
            emb = F.normalize(bert.encode(texts).cpu() - cache.mu, p=2, dim=-1)
            cos.append(F.cosine_similarity(emb, cache.test[idx], dim=-1))
        c = torch.cat(cos)
        res[str(snr)] = {"cos": c.mean().item(), "se": (c.std() / math.sqrt(len(c))).item(),
                         "token_err": n_err / max(n_tok, 1), "n_users": len(c)}
        print(f"U={U} T={T} L={L} snr={snr} cos={res[str(snr)]['cos']:.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te = load_agnews_labeled()
    bert = Extractor("bert-base-uncased", dev)
    cache = SplitCache(bert, tr, te)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for c in args.configs:
        U, T, L = map(int, c.split(","))
        assert T * L == 3072
        t0 = time.time()
        out[f"U{U}_T{T}_L{L}"] = ev(bert, cache, dev, U, T, L, args.frames)
        out[f"U{U}_T{T}_L{L}"]["seconds"] = time.time() - t0
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
