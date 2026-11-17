# =========================================================
# emb_captions.py — BERT sentence embeddings of image captions, five per image (2026-11-16).
# Shared content: the captions of one image are written by different people who view the same scene,
# so they stand for users (cameras, observers) that describe one scene. Same encoder and preprocessing as
# the AG News embeddings of the letter (rev_lib.Extractor: frozen BERT-base uncased, mean pooling over
# tokens, max 64 tokens; centering by the training-pool mean and l2 normalization, eq. (1)).
#   coco     yerevann/coco-karpathy (train parquet, 82,783 images): images are shuffled with seed 0;
#            the first 8,000 images form the training pool, the next 2,000 the test images
#   flickr   nlphuji/flickr30k annotations: 8,000 images of the train split as training pool, the
#            1,000 test images as test images
# Output ../data/emb_<dataset>_caps_bert-base-uncased.pt with
#   train (8000*5, 768), test (n_test, 5, 768), train_img / test_img ids, texts, mu
# Usage: python emb_captions.py --dataset coco
# =========================================================
import argparse, glob, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
from rev_lib import Extractor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def load(dataset, n_train, n_test):
    if dataset == "coco":
        import pyarrow.parquet as pq
        f = glob.glob(os.path.join(HUB, "datasets--yerevann--coco-karpathy/snapshots/*/data/train-*.parquet"))[0]
        rows = [(r["cocoid"], list(r["sentences"])) for r in pq.read_table(f, columns=["cocoid", "sentences"]).to_pylist()]
        rows = [r for r in rows if len(r[1]) >= 5]
        random.Random(0).shuffle(rows)
        return rows[:n_train], rows[n_train:n_train + n_test]
    import pandas as pd
    d = pd.read_csv(glob.glob(os.path.join(HUB, "datasets--nlphuji--flickr30k/snapshots/*/flickr_annotations_30k.csv"))[0])
    rows = {s: [(int(r.img_id), json.loads(r.raw)) for r in d[d["split"] == s].itertuples()] for s in ("train", "test")}
    tr = [r for r in rows["train"] if len(r[1]) >= 5]
    random.Random(0).shuffle(tr)
    return tr[:n_train], [r for r in rows["test"] if len(r[1]) >= 5][:n_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = load(a.dataset, a.n_train, a.n_test)
    ext = Extractor(a.model, dev)
    tr_txt = [c.strip() for _, caps in tr for c in caps[:5]]
    te_txt = [c.strip() for _, caps in te for c in caps[:5]]
    E_tr = ext.encode(tr_txt)
    E_te = ext.encode(te_txt)
    mu = E_tr.mean(0, keepdim=True)
    C = {"train": F.normalize(E_tr - mu, dim=-1), "test": F.normalize(E_te - mu, dim=-1).reshape(len(te), 5, -1),
         "train_img": [i for i, _ in tr], "test_img": [i for i, _ in te], "train_texts": tr_txt, "test_texts": te_txt,
         "mu": mu, "dataset": a.dataset, "model": a.model}
    out = os.path.join(DATA, f"emb_{a.dataset}_caps_{a.model.replace('/', '_')}.pt")
    torch.save(C, out)
    T = C["test"].double()
    G = torch.einsum("ncd,ned->nce", T, T)                      # cosine of caption pairs of one image
    off = G[:, ~torch.eye(5, dtype=bool)].mean().item()
    R = T[torch.randperm(len(te))[:len(te)], 0] @ T[:, 1].T
    print(f"wrote {out}: train {tuple(C['train'].shape)} test {tuple(C['test'].shape)}; mean CosSim of two captions "
          f"of one image {off:.3f}, of different images {R.diagonal().mean().item():.3f}", flush=True)


if __name__ == "__main__":
    main()
