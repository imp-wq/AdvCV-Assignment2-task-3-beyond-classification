# -*- coding: utf-8 -*-
"""
Train linear probes on first-token features.

Inputs (created by extract_first_token_logits.py):
  out/features_{split}_{cls}.npz
    - X: [N, D]  projected first-token logits
    - y: [N]     binary labels for this class
    - x_meta: [N,5] = [image_id, x1, y1, x2, y2]
    - img_files: [N] image file names
    - classes, proj_dim, vocab_dim, model

Outputs:
  out/probes/{cls}.joblib         # scaler+logreg
  out/probes/{cls}_threshold.json # calibrated threshold on val
  out/preds_test_{cls}.jsonl      # per-candidate predictions on test
  Console metrics (AP/ROC-AUC/F1/... & CorLoc@1/3)
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, precision_score, recall_score
)
import joblib

# ----------------------------- Utils -----------------------------

def load_feats(split:str, cls:str, root="out"):
    path = Path(root) / f"features_{split}_{cls}.npz"
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int32)
    x_meta = z["x_meta"].astype(np.int64)   # [img_id, x1,y1,x2,y2]
    img_files = z["img_files"]
    return X, y, x_meta, img_files

def choose_threshold_by_f1(y_true, scores):
    """Return threshold that maximizes F1 on (y, scores)."""
    ps, rs, ths = precision_recall_curve(y_true, scores)  # ths len = n-1
    # compute F1 for each threshold point (skip first ps/rs element which corresponds to threshold=None)
    f1s = 2 * ps[:-1] * rs[:-1] / (ps[:-1] + rs[:-1] + 1e-12)
    if f1s.size == 0:
        return 0.5, 0.0
    idx = np.nanargmax(f1s)
    return float(ths[idx]), float(f1s[idx])

def group_indices_by_image(x_meta):
    img_ids = x_meta[:,0].tolist()
    groups = defaultdict(list)
    for i, gid in enumerate(img_ids):
        groups[int(gid)].append(i)
    return groups

def corloc_at_k(y_true, scores, x_meta, topk=1):
    """
    CorLoc@K on image level:
      - for each image, sort candidates by score desc
      - success if any of the top-K candidates has y=1
    """
    groups = group_indices_by_image(x_meta)
    ok = 0
    for _, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        idxs = idxs[:topk]
        hit = any(int(y_true[i]) == 1 for i in idxs)
        ok += int(hit)
    return ok / max(1, len(groups))

def eval_metrics(y_true, scores, thr, x_meta):
    """Compute a bundle of metrics."""
    # Probabilistic metrics
    ap = average_precision_score(y_true, scores) if len(np.unique(y_true))>1 else float('nan')
    try:
        auc = roc_auc_score(y_true, scores) if len(np.unique(y_true))>1 else float('nan')
    except Exception:
        auc = float('nan')

    # Thresholded metrics
    y_pred = (scores >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # Image-level CorLoc
    cor1 = corloc_at_k(y_true, scores, x_meta, topk=1)
    cor3 = corloc_at_k(y_true, scores, x_meta, topk=3)
    return dict(AP=ap, ROC_AUC=auc, F1=f1, Precision=prec, Recall=rec, CorLoc@1=cor1, CorLoc@3=cor3)

def save_preds_jsonl(path, scores, y_true, x_meta, img_files):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, y, m, fn in zip(scores, y_true, x_meta, img_files):
            rec = {
                "image_id": int(m[0]),
                "bbox": [int(m[1]), int(m[2]), int(m[3]), int(m[4])],
                "score": float(s),
                "label": int(y),
                "file": str(fn)
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_row(title, metrics:dict):
    keys = ["AP","ROC_AUC","F1","Precision","Recall","CorLoc@1","CorLoc@3"]
    vals = [metrics.get(k, float('nan')) for k in keys]
    msg = f"{title:>10s} | " + " | ".join([f"{k} {v:6.4f}" if isinstance(v,(int,float)) and not np.isnan(v) else f"{k}  n/a" for k,v in zip(keys,vals)])
    print(msg)

# ----------------------------- Main -----------------------------

def run_for_class(cls:str, train_split:str, val_split:str, test_split:str,
                  C:float, max_iter:int, out_dir="out", verbose=True):
    # 1) Load data
    Xtr, ytr, mtr, _ = load_feats(train_split, cls, root=out_dir)
    Xva, yva, mva, _ = load_feats(val_split, cls, root=out_dir)
    Xte, yte, mte, fte = load_feats(test_split, cls, root=out_dir)

    if verbose:
        print(f"[{cls}] Shapes: Xtr {Xtr.shape}, Xva {Xva.shape}, Xte {Xte.shape}")

    # 2) Pipeline: StandardScaler + LogisticRegression
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            C=C, max_iter=max_iter, class_weight="balanced", n_jobs=None, solver="liblinear"
        ))
    ])
    pipe.fit(Xtr, ytr)

    # 3) Scores
    def scores_of(X):
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "decision_function"):
            s = pipe.decision_function(X)
        else:
            s = pipe.predict_proba(X)[:,1]
        return s.astype(np.float32)

    s_tr = scores_of(Xtr)
    s_va = scores_of(Xva)
    s_te = scores_of(Xte)

    # 4) Calibrate threshold on val by max F1
    thr, best_f1 = choose_threshold_by_f1(yva, s_va)
    if verbose:
        print(f"[{cls}] Best threshold on val: {thr:.6f} (F1={best_f1:.4f})")

    # 5) Metrics
    mt_tr = eval_metrics(ytr, s_tr, thr, mtr)
    mt_va = eval_metrics(yva, s_va, thr, mva)
    mt_te = eval_metrics(yte, s_te, thr, mte)

    if verbose:
        log_row("Train", mt_tr)
        log_row("Val",   mt_va)
        log_row("Test",  mt_te)

    # 6) Save model & threshold
    probes_dir = Path(out_dir) / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, probes_dir / f"{cls}.joblib")
    with open(probes_dir / f"{cls}_threshold.json", "w") as f:
        json.dump({"threshold": thr}, f)

    # 7) Save test predictions
    save_preds_jsonl(Path(out_dir)/f"preds_test_{cls}.jsonl", s_te, yte, mte, fte)

    return mt_tr, mt_va, mt_te

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, nargs="*", default=["person","car","dog","bicycle","chair"],
                        help="Which classes to train. Default 5 classes.")
    parser.add_argument("--train", type=str, default="train", help="Train split name")
    parser.add_argument("--val",   type=str, default="val",   help="Val split name")
    parser.add_argument("--test",  type=str, default="test",  help="Test split name")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg regularization inverse strength")
    parser.add_argument("--max_iter", type=int, default=1000, help="LogReg max_iter")
    parser.add_argument("--out_dir", type=str, default="out")
    args = parser.parse_args()

    all_te = []
    print("==== Linear Probe Training ====")
    print(f"Classes: {args.classes}")

    for cls in args.classes:
        print("\n" + "="*20 + f" {cls} " + "="*20)
        _, _, mt_te = run_for_class(
            cls, args.train, args.val, args.test,
            C=args.C, max_iter=args.max_iter, out_dir=args.out_dir, verbose=True
        )
        all_te.append(mt_te)

    # Macro average across classes (on test)
    keys = ["AP","ROC_AUC","F1","Precision","Recall","CorLoc@1","CorLoc@3"]
    avg = {k: float(np.nanmean([m[k] for m in all_te])) for k in keys}
    print("\n" + "-"*12 + " Macro-Average (Test) " + "-"*12)
    log_row("Average", avg)

if __name__ == "__main__":
    main()
