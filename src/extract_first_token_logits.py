import os, json, math, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForCausalLM

# ----------------------------- Config -----------------------------
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

COCO_ROOT = "data/coco"            # 与上一步一致
SAMPLES_JSONL = "out/samples.jsonl"
LABELS_NPZ_TMPL = "out/labels_{split}.npz"

# 模型（3B，显存友好）
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# 批大小：根据你的显存调整，8/16/32 都可尝试
BATCH_SIZE = 16

# 降维到 1024（固定随机投影，保证各 split 一致）
PROJ_DIM = 1024
PROJ_PATH = Path("out/proj_matrix.npz")   # 保存/加载随机投影矩阵

# 5 个类（与前一步一致）
CLASSES = ["person", "car", "dog", "bicycle", "chair"]

# 提示模板（英文更稳，Yes/No 输出规范）
PROMPT_TMPL = "In this box, is there a {cls}? Answer Yes or No."

# 输出
OUT_FEAT_TMPL = "out/features_{split}.npz"

# ----------------------------- Utils -----------------------------
def load_split(split):
    """从 labels_{split}.npz 恢复样本索引、bbox 与标签；并对齐 samples.jsonl 的顺序。"""
    meta = np.load(LABELS_NPZ_TMPL.format(split=split), allow_pickle=True)
    X_meta = meta["x_meta"]     # [N, 5] = [image_id, x1,y1,x2,y2]
    Y = meta["y"]               # [N, 5]
    IMG_FILES = meta["img_files"]  # [N]
    # 直接返回（顺序即是 jsonl 写入顺序）
    return X_meta, Y, IMG_FILES

def crop_box(img_path, box):
    x1,y1,x2,y2 = map(int, box)
    im = Image.open(img_path).convert("RGB")
    w,h = im.size
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2<=x1+1 or y2<=y1+1:
        # 退化框，给个最小补偿
        x2 = min(w, x1+2); y2 = min(h, y1+2)
    return im.crop((x1,y1,x2,y2))

@torch.inference_mode()
def next_token_logits(model, processor, images, texts, device):
    """
    对一批 (image, text) 获取“下一 token（首个生成 token）”的 logits。
    返回：list[np.ndarray]，每个是 [vocab_size] 的向量
    """
    inputs = processor(images=images, text=texts, return_tensors="pt").to(device)
    # 对于 decoder-only 模型，最后一个位置的 logits 即下一 token 分布
    out = model(**inputs)
    # out.logits: [B, T, V]
    logits = out.logits[:, -1, :]  # [B, V]
    return [x.detach().cpu().float().numpy() for x in logits.unbind(dim=0)]

def build_or_load_proj(vocab_dim, out_dim=1024, path=PROJ_PATH):
    """
    构造固定随机高斯投影矩阵 R:[vocab_dim, out_dim] 并保存/加载。
    使用 1/sqrt(out_dim) 的缩放，保近似内积。
    """
    path = Path(path)
    if path.exists():
        z = np.load(path)
        R = z["R"]
        if R.shape != (vocab_dim, out_dim):
            raise ValueError(f"Existing proj shape {R.shape} != ({vocab_dim},{out_dim})")
        return R
    # 生成稀疏或稠密的高斯矩阵（这里用稠密，简单可靠）
    R = np.random.normal(0.0, 1.0, size=(vocab_dim, out_dim)).astype(np.float32)
    R *= (1.0/ math.sqrt(out_dim))
    np.savez_compressed(path, R=R)
    print(f"[Info] Saved projection matrix to {path} with shape {R.shape}")
    return R

def make_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield slice(i, i+bs)

# ----------------------------- Main -----------------------------
def main(split):
    assert split in ("train","val","test")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Info] Loading model: {MODEL_NAME} on {device}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    # 载入样本（与上一步顺序一致）
    X_meta, Y, IMG_FILES = load_split(split)
    N = X_meta.shape[0]
    print(f"[Info] Split={split} samples: {N}")

    # 预跑一个样本，拿到词表维度
    test_img = Image.open(os.path.join(COCO_ROOT, "val2017", IMG_FILES[0])).convert("RGB")
    test_prompt = PROMPT_TMPL.format(cls=CLASSES[0])
    test_logits = next_token_logits(model, processor, [test_img], [test_prompt], device)[0]
    V = test_logits.shape[0]
    print(f"[Info] Vocab dim = {V}")

    # 随机投影矩阵（固定保存，保证各 split 一致）
    R = build_or_load_proj(V, PROJ_DIM, PROJ_PATH)   # [V, D]
    D = R.shape[1]

    # 输出特征矩阵（N, D）与标签（N, C）
    X_proj = np.zeros((N, D), dtype=np.float32)
    Y_mat = Y.astype(np.int8)

    # 为每个样本构造文本，并裁剪框
    coco_img_root = os.path.join(COCO_ROOT, "val2017")

    # 批处理
    for batch_idx in tqdm(list(make_batches(range(N), BATCH_SIZE)), desc=f"Extract {split}"):
        idxs = range(batch_idx.start, min(batch_idx.stop, N))
        images, texts = [], []
        for i in idxs:
            img_file = IMG_FILES[i]
            _, x1,y1,x2,y2 = X_meta[i]
            img_path = os.path.join(coco_img_root, img_file)
            crop = crop_box(img_path, (x1,y1,x2,y2))
            # 这里我们一次只问一个类：将 5 类分别抽一次特征会 5x 成本。
            # 为了省算力，我们让特征与“类无关”（仅图像框+通用问题），但这会略降精度。
            # —— 推荐：为每个类各抽一次（更准）；若算力紧张，先做“通用特征版本”打通流程。
            # 这里默认做“每类一次”，因此我们需要为每条样本抽 5 次并拼接。若想省算力，改为一个通用 prompt。
            pass
        # 我们改为“每类一次”，但为了内存，我们逐类循环再写入矩阵。
        # 所以这里暂时什么都不做，此处仅占位。
        break

    # --------- 更准确的实现：逐类提取，再拼接到 (N, D*C) 或分别保存每类一份 ----------
    # 出于可解释性与与第3步（每类一分类器）的一致性，我们保存“每类一个特征矩阵”：
    #   out/features_{split}_{cls}.npz: X:[N,D], y:[N], x_meta:[N,5], img_files
    for ci, cls in enumerate(CLASSES):
        print(f"[Info] Extracting class={cls}")
        Xc = np.zeros((N, D), dtype=np.float32)
        # 分批
        for batch_idx in tqdm(list(make_batches(range(N), BATCH_SIZE)), desc=f"{split}:{cls}"):
            idxs = range(batch_idx.start, min(batch_idx.stop, N))
            images, texts = [], []
            for i in idxs:
                img_file = IMG_FILES[i]
                _, x1,y1,x2,y2 = X_meta[i]
                img_path = os.path.join(coco_img_root, img_file)
                crop = crop_box(img_path, (x1,y1,x2,y2))
                images.append(crop)
                texts.append(PROMPT_TMPL.format(cls=cls))
            logits_list = next_token_logits(model, processor, images, texts, device) # list of [V]
            # 投影到 D 维
            # logits_list 是长度 len(idxs) 的列表，每个向量 [V]，我们做 [V] @ [V,D] -> [D]
            # 为了效率，堆叠成矩阵再一次性点乘
            L = np.stack(logits_list, axis=0).astype(np.float32)   # [B, V]
            Xc_batch = L @ R   # [B, D]
            Xc[batch_idx] = Xc_batch

        yc = Y_mat[:, ci].astype(np.int8)  # 该类标签
        out_path = OUT_FEAT_TMPL.format(split=split).replace(".npz", f"_{cls}.npz")
        np.savez_compressed(
            out_path,
            X=Xc, y=yc, x_meta=X_meta, img_files=IMG_FILES, classes=np.array(CLASSES),
            proj_dim=np.array([D]), vocab_dim=np.array([V]), model=np.array([MODEL_NAME])
        )
        print(f"[OK] Saved {out_path} | X {Xc.shape}, y {yc.shape}")

    print("[Done] All classes extracted.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    args = parser.parse_args()
    main(args.split)
