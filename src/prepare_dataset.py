import json, random, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from skimage import io

# ----------------------------- Config -----------------------------
SEED = 2025
random.seed(SEED); np.random.seed(SEED)

COCO_ROOT = "data/coco"   # <<< 改成你的路径
ANN = f"{COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR = f"{COCO_ROOT}/val2017"

# 5 个目标类
CLASSES = ["person", "car", "dog", "bicycle", "chair"]

# 每类最多抽多少张图（图像级去重后通常 ~500-700 张）
MAX_IMGS_PER_CLASS = 100

# 候选框配置：网格 + 随机
GRID = 4  # 4x4 网格；若想更密，改为 5
N_RANDOM = 8  # 每图额外随机框数
MIN_BOX_FRAC = 0.10  # 随机框短边相对 min(H,W) 的最小比例

# IoU 正样本阈值
POS_IOU_THR = 0.5

# 切分比例（按图像级）
SPLIT_RATIOS = (0.8, 0.1, 0.1) # train/val/test

OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Utils -----------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def coco_xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]

def clip_box(x1, y1, x2, y2, W, H):
    return [max(0, min(x1, W-1)),
            max(0, min(y1, H-1)),
            max(0, min(x2, W-1)),
            max(0, min(y2, H-1))]

def gen_grid_boxes(W, H, grid=4):
    boxes = []
    xs = np.linspace(0, W, grid+1, dtype=int)
    ys = np.linspace(0, H, grid+1, dtype=int)
    for i in range(grid):
        for j in range(grid):
            x1, y1 = xs[i], ys[j]
            x2, y2 = xs[i+1], ys[j+1]
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                boxes.append([x1, y1, x2, y2])
    return boxes

def gen_random_boxes(W, H, n=8, min_frac=0.10):
    boxes = []
    short = min(W, H)
    min_sz = max(2, int(short * min_frac))
    for _ in range(n):
        w = random.randint(min_sz, W)
        h = random.randint(min_sz, H)
        if w >= W: w = W-1
        if h >= H: h = H-1
        x1 = random.randint(0, W - w)
        y1 = random.randint(0, H - h)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append([x1, y1, x2, y2])
    return boxes

# ----------------------------- Load COCO -----------------------------
coco = COCO(ANN)
# map class name -> cat_id
name_to_cid = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
class_cids = [name_to_cid[n] for n in CLASSES]

# ----------------------------- Select images -----------------------------
# 收集每类的图片，然后取并集，限制每类最多 MAX_IMGS_PER_CLASS
selected_img_ids = set()
per_class_imgs = {}
for cname, cid in zip(CLASSES, class_cids):
    img_ids = coco.getImgIds(catIds=[cid])
    random.shuffle(img_ids)
    keep = img_ids[:MAX_IMGS_PER_CLASS]
    per_class_imgs[cname] = set(keep)
    selected_img_ids.update(keep)

selected_img_ids = list(selected_img_ids)
random.shuffle(selected_img_ids)

# ----------------------------- Build GT index per image & class -----------------------------
# 为加速 IoU 计算，把每张图每个目标类的 GT 框整理出来
gt_bboxes = defaultdict(lambda: defaultdict(list))  # gt_bboxes[img_id][class_name] = [xyxy...]
imgid_to_meta = {}
for img_id in selected_img_ids:
    img_info = coco.loadImgs([img_id])[0]
    W, H = img_info["width"], img_info["height"]
    imgid_to_meta[img_id] = {"file_name": img_info["file_name"], "W": W, "H": H}
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        if ann.get("iscrowd", 0) == 1:  # 简化：忽略 crowd
            continue
        cid = ann["category_id"]
        cname = None
        for n, c in zip(CLASSES, class_cids):
            if c == cid:
                cname = n
                break
        if cname is None:
            continue
        x, y, w, h = ann["bbox"]
        box = coco_xywh_to_xyxy(x, y, w, h)
        # clip 防越界
        box = clip_box(*box, W, H)
        gt_bboxes[img_id][cname].append(box)

# ----------------------------- Image-level split -----------------------------
n = len(selected_img_ids)
n_train = int(n * SPLIT_RATIOS[0])
n_val   = int(n * SPLIT_RATIOS[1])
train_ids = selected_img_ids[:n_train]
val_ids   = selected_img_ids[n_train:n_train+n_val]
test_ids  = selected_img_ids[n_train+n_val:]
splits = {"train": train_ids, "val": val_ids, "test": test_ids}
json.dump(splits, open(OUT_DIR/"splits.json", "w"))

print(f"Total images: {n} | train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}")

# ----------------------------- Generate candidates & labels -----------------------------
# 输出到 samples.jsonl：一行一个候选框样本
# 字段：{image_id, split, file_name, W,H, bbox:[x1,y1,x2,y2], labels:{cls:0/1,...}}
out_jsonl = open(OUT_DIR/"samples.jsonl", "w", encoding="utf-8")
counts = {"total_samples": 0, "pos_per_class": {c:0 for c in CLASSES}, "neg_per_class": {c:0 for c in CLASSES}}

for split, img_ids in splits.items():
    for img_id in tqdm(img_ids, desc=f"Building {split}"):
        meta = imgid_to_meta[img_id]
        W, H = meta["W"], meta["H"]
        # 候选框
        boxes = gen_grid_boxes(W, H, GRID) + gen_random_boxes(W, H, N_RANDOM, MIN_BOX_FRAC)

        # 为该图计算每个候选对每个类的标签
        for box in boxes:
            labels = {}
            for cname in CLASSES:
                gt_list = gt_bboxes[img_id].get(cname, [])
                # max IoU over all GTs of this class
                mx = 0.0
                for g in gt_list:
                    mx = max(mx, iou_xyxy(box, g))
                y = 1 if mx >= POS_IOU_THR else 0
                labels[cname] = y
                counts["pos_per_class"][cname] += int(y==1)
                counts["neg_per_class"][cname] += int(y==0)
            rec = {
                "image_id": int(img_id),
                "split": split,
                "file_name": meta["file_name"],
                "W": int(W), "H": int(H),
                "bbox": [int(v) for v in box],
                "labels": labels
            }
            out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["total_samples"] += 1

out_jsonl.close()
print("Stats:", json.dumps(counts, indent=2))

# ----------------------------- (Optional) Dense matrices -----------------------------
# 方便后续训练时快速载入：把每个 split 的样本堆叠成矩阵（仅标签与坐标/索引），特征第二步再抽
def build_npz(split_name):
    X_meta = []  # [img_id, x1,y1,x2,y2]
    Y = []       # [N, C]
    IMG = []
    with open(OUT_DIR/"samples.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != split_name: continue
            X_meta.append([r["image_id"], *r["bbox"]])
            Y.append([r["labels"][c] for c in CLASSES])
            IMG.append(r["file_name"])
    if len(X_meta)==0:
        return
    X_meta = np.array(X_meta, dtype=np.int64)
    Y = np.array(Y, dtype=np.int8)
    np.savez_compressed(OUT_DIR/f"labels_{split_name}.npz",
                        x_meta=X_meta, y=Y, img_files=np.array(IMG),
                        classes=np.array(CLASSES))
    print(f"Saved {split_name}: x_meta {X_meta.shape}, y {Y.shape}")

for sp in ["train","val","test"]:
    build_npz(sp)
