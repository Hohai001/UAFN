import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


BACKBONE_TO_MODEL = {
    "macbert": "hfl/chinese-macbert-large",
    "bert": "bert-base-chinese",
}

BACKBONE_TO_OUTPUT = {
    "macbert": "text_features_v2_macbert_large.h5",
    "bert": "text_features_bert_base_chinese.h5",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Extract CH-SIMS text features from MacBERT/BERT")
    parser.add_argument("--data_path", type=str, default="/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv")
    parser.add_argument("--text_backbone", type=str, choices=["macbert", "bert"], default="macbert")
    parser.add_argument("--model_name", type=str, default=None, help="Optional manual model override")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path override")
    parser.add_argument("--max_length", type=int, default=128)
    return parser


def extract_text_features(args):
    model_name = args.model_name or BACKBONE_TO_MODEL[args.text_backbone]
    output_path = Path(args.output_path or BACKBONE_TO_OUTPUT[args.text_backbone])
    data_path = Path(args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not data_path.exists():
        raise FileNotFoundError(f"Meta file not found: {data_path}")

    df = pd.read_csv(data_path)
    texts = df["text"].tolist()
    video_ids = df["video_id"].astype(str).tolist()
    sample_keys = [f"{vid.replace('/', '_')}_{i}" for i, vid in enumerate(video_ids)]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"开始提取文本特征：backbone={args.text_backbone}, model={model_name}, device={device}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        for i, text in enumerate(tqdm(texts)):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                last_4 = torch.stack(outputs.hidden_states[-4:], dim=0)
                avg_hidden = last_4.mean(dim=0)

                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).float()
                h_t = (avg_hidden * mask).sum(dim=1) / mask.sum(dim=1)
                h_t = h_t.squeeze().cpu().numpy().astype(np.float32)

            h5f.create_dataset(sample_keys[i], data=h_t)

    print(f"提取完成！保存路径: {output_path}")


if __name__ == "__main__":
    parser = build_parser()
    extract_text_features(parser.parse_args())
