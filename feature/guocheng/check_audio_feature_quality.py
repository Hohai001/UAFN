import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quality checks for audio features stored in an HDF5 file."
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv",
        help="Path to meta.csv used to build sample keys.",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default="audio_features.h5",
        help="Path to audio feature HDF5 file.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="video_id",
        help="Column used to build sample keys.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="annotation",
        help="Label column for separability checks. Set empty string to disable.",
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default="mode",
        help="Split column for train/valid/test evaluation.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=768,
        help="Expected feature dimension.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=10,
        help="Max number of keys to print in each issue list.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional output path to save the full report JSON.",
    )
    return parser.parse_args()


def build_sample_keys(df: pd.DataFrame, id_col: str):
    ids = df[id_col].astype(str).tolist()
    # Keep exactly the same key rule used in extract_sims_wav2vec.py.
    return [f"{vid.replace('/', '_')}_{i}" for i, vid in enumerate(ids)]


def macro_f1(y_true, y_pred, classes):
    f1_scores = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def l2_normalize(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    return x / safe_norms


def centroid_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
):
    classes = np.array(sorted(pd.unique(y_train)))
    if classes.size < 2 or x_eval.shape[0] == 0:
        return None

    x_train_n = l2_normalize(x_train)
    x_eval_n = l2_normalize(x_eval)

    centroids = []
    valid_classes = []
    for c in classes:
        mask = y_train == c
        if np.sum(mask) == 0:
            continue
        center = x_train_n[mask].mean(axis=0, keepdims=True)
        center = l2_normalize(center)[0]
        centroids.append(center)
        valid_classes.append(c)

    if not centroids:
        return None

    centroid_mat = np.stack(centroids, axis=1)  # [dim, C]
    sims = x_eval_n @ centroid_mat
    pred_idx = np.argmax(sims, axis=1)
    y_pred = np.array([valid_classes[i] for i in pred_idx], dtype=object)

    y_eval_obj = y_eval.astype(object)
    acc = float(np.mean(y_pred == y_eval_obj))
    f1 = macro_f1(y_eval_obj, y_pred, np.array(valid_classes, dtype=object))
    return {"accuracy": acc, "macro_f1": f1, "num_classes": int(len(valid_classes))}


def print_issue_list(title: str, values, max_print: int):
    print(f"{title}: {len(values)}")
    if values:
        preview = values[:max_print]
        print(f"  sample -> {preview}")


def main():
    args = parse_args()
    meta_path = Path(args.meta_path)
    h5_path = Path(args.h5_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"h5 file not found: {h5_path}")

    df = pd.read_csv(meta_path)
    if args.id_col not in df.columns:
        raise ValueError(f"id column '{args.id_col}' not found in meta.csv")

    expected_keys = build_sample_keys(df, args.id_col)
    expected_set = set(expected_keys)

    report = {
        "paths": {"meta": str(meta_path), "h5": str(h5_path)},
        "meta_rows": int(len(df)),
        "expected_dim": int(args.expected_dim),
    }

    with h5py.File(h5_path, "r") as h5f:
        h5_keys = list(h5f.keys())
        h5_key_set = set(h5_keys)

        missing_keys = [k for k in expected_keys if k not in h5_key_set]
        extra_keys = [k for k in h5_keys if k not in expected_set]

        good_row_idx = []
        good_keys = []
        vectors = []
        wrong_dim_keys = []
        bad_shape_keys = []

        for i, key in enumerate(expected_keys):
            if key not in h5_key_set:
                continue
            v = np.asarray(h5f[key])
            if v.ndim != 1:
                bad_shape_keys.append(key)
                continue
            if args.expected_dim > 0 and v.shape[0] != args.expected_dim:
                wrong_dim_keys.append((key, int(v.shape[0])))
                continue
            good_row_idx.append(i)
            good_keys.append(key)
            vectors.append(v.astype(np.float32, copy=False))

    x = np.stack(vectors, axis=0) if vectors else np.empty((0, args.expected_dim), dtype=np.float32)
    aligned_df = df.iloc[good_row_idx].copy()

    report["alignment"] = {
        "h5_keys": int(len(h5_keys)),
        "matched_rows": int(len(good_row_idx)),
        "coverage_ratio": float(len(good_row_idx) / max(len(df), 1)),
        "missing_keys": int(len(missing_keys)),
        "extra_keys": int(len(extra_keys)),
        "bad_shape_keys": int(len(bad_shape_keys)),
        "wrong_dim_keys": int(len(wrong_dim_keys)),
    }

    print("=== Alignment Check ===")
    print(f"meta rows: {len(df)}")
    print(f"h5 keys: {len(h5_keys)}")
    print(f"matched rows: {len(good_row_idx)}")
    print(f"coverage ratio: {len(good_row_idx) / max(len(df), 1):.4f}")
    print_issue_list("missing keys", missing_keys, args.max_print)
    print_issue_list("extra keys", extra_keys, args.max_print)
    print_issue_list("bad-shape keys", bad_shape_keys, args.max_print)
    print_issue_list("wrong-dim keys", wrong_dim_keys, args.max_print)

    print("\n=== Vector Health Check ===")
    if x.shape[0] == 0:
        print("No valid vectors to inspect.")
        report["vector_health"] = {"num_vectors": 0}
    else:
        finite_mask = np.isfinite(x).all(axis=1)
        nan_inf_rows = np.where(~finite_mask)[0]

        norms = np.linalg.norm(x, axis=1)
        zero_rows = np.where(norms < 1e-12)[0]

        health = {
            "num_vectors": int(x.shape[0]),
            "dim": int(x.shape[1]),
            "nan_inf_rows": int(len(nan_inf_rows)),
            "zero_rows": int(len(zero_rows)),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "feat_mean_abs": float(np.mean(np.abs(x))),
            "feat_std_mean": float(np.mean(np.std(x, axis=0))),
        }
        report["vector_health"] = health

        for k, v in health.items():
            print(f"{k}: {v}")
        if len(nan_inf_rows) > 0:
            bad_keys = [good_keys[i] for i in nan_inf_rows[: args.max_print]]
            print(f"nan/inf sample keys -> {bad_keys}")
        if len(zero_rows) > 0:
            z_keys = [good_keys[i] for i in zero_rows[: args.max_print]]
            print(f"zero-vector sample keys -> {z_keys}")

    label_col = (args.label_col or "").strip()
    can_eval = (
        x.shape[0] > 0
        and label_col
        and label_col in aligned_df.columns
        and args.split_col in aligned_df.columns
    )

    print("\n=== Separability (Nearest Centroid) ===")
    if not can_eval:
        print("Skipped: require valid vectors + label_col + split_col.")
        report["separability"] = {"skipped": True}
    else:
        y = aligned_df[label_col].astype(str).values
        split = aligned_df[args.split_col].astype(str).str.lower().values

        train_mask = split == "train"
        valid_mask = np.isin(split, ["valid", "val", "dev"])
        test_mask = split == "test"

        if np.sum(train_mask) == 0:
            print("Skipped: no train split rows found.")
            report["separability"] = {"skipped": True, "reason": "no_train_rows"}
        else:
            sep = {"label_col": label_col, "split_col": args.split_col}

            if np.sum(valid_mask) > 0:
                valid_out = centroid_eval(
                    x[train_mask], y[train_mask], x[valid_mask], y[valid_mask]
                )
                sep["valid"] = valid_out
                if valid_out is not None:
                    print(
                        f"valid -> acc={valid_out['accuracy']:.4f}, "
                        f"macro_f1={valid_out['macro_f1']:.4f}, "
                        f"classes={valid_out['num_classes']}"
                    )
            if np.sum(test_mask) > 0:
                test_out = centroid_eval(
                    x[train_mask], y[train_mask], x[test_mask], y[test_mask]
                )
                sep["test"] = test_out
                if test_out is not None:
                    print(
                        f"test  -> acc={test_out['accuracy']:.4f}, "
                        f"macro_f1={test_out['macro_f1']:.4f}, "
                        f"classes={test_out['num_classes']}"
                    )

            if "valid" not in sep and "test" not in sep:
                print("Skipped: no valid/test rows found.")
                sep["skipped"] = True
                sep["reason"] = "no_eval_rows"

            report["separability"] = sep

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport JSON saved to: {out}")


if __name__ == "__main__":
    main()
