## Project Structure
```
project/
  data/
    coco/
      val2017/                     # COCO val2017 图像（原样）
      annotations/instances_val2017.json
  out/
    splits.json                    # 图像级 train/val/test 切分
    samples.jsonl                  # 样本级（每个候选框一行）
    labels.npz                     # 稀疏/稠密矩阵化标签（可选）
  src/
    build_dataset.py
```

## 1. Download Coco Dataset
```sh
sh download_dataset.sh
```
## 2. Prepare Dataset
```sh
python ./src/prepare_dataset.py
```
## 3. Extract First Token
```sh
# 训练集
python ./src/extract_first_token_logits.py --split train
# 验证集
python ./src/extract_first_token_logits.py --split val
# 测试集
python ./src/extract_first_token_logits.py --split test
```

## 4. Train Probe
```sh
# 全部五类（默认）
python src/train_probe.py

# 只训练/评测某一类
python src/train_probe.py --classes car

# 调参（正则系数C）
python src/train_probe.py --C 0.5
```

