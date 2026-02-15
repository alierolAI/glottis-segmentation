from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt


GT_ROOT_CANDIDATES = [
    Path("/kaggle/working/Test50_fixed3d/labelsTs"),
    Path("/kaggle/working/Test50_fixed3d/labelsTr"),
    Path("/kaggle/input/testing/nnUNet_raw/Dataset102_BAGLS2D_TEST50/labelsTs"),
    Path("/kaggle/input/testing/nnUNet_raw/Dataset102_BAGLS2D_TEST50/labelsTr"),
]
PRED_ROOT_CANDIDATES = [
    Path("/kaggle/working/preds_test50"),
    Path("/kaggle/working/preds_test50_fixed"),
    Path("/kaggle/working/preds_test50_roi"),
    Path("/kaggle/working/preds_test50_3d"),
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_nii(path: Path):
    nii = nib.load(str(path))
    arr = np.squeeze(nii.get_fdata())
    return arr, nii.header.get_zooms()


def to_binary(arr):
    if arr.dtype.kind in ("f", "c"):
        return (arr > 0.5).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def surface_mask(bin_mask: np.ndarray):
    if bin_mask.ndim == 2:
        struct = np.ones((3, 3), dtype=bool)
    else:
        struct = np.ones((3, 3, 3), dtype=bool)
    er = binary_erosion(bin_mask.astype(bool), structure=struct, border_value=0)
    surf = bin_mask.astype(bool) & (~er)
    return surf


def surface_distances(a: np.ndarray, b: np.ndarray, spacing):
    sa = surface_mask(a)
    sb = surface_mask(b)

    if sb.sum() == 0 or sa.sum() == 0:
        return np.array([], dtype=np.float32)

    dt = distance_transform_edt(~sb, sampling=spacing[:sb.ndim])
    return dt[sa].astype(np.float32)


def hd95_asd(gt: np.ndarray, pr: np.ndarray, spacing):
    gt = gt.astype(np.uint8)
    pr = pr.astype(np.uint8)

    gt_sum = int(gt.sum())
    pr_sum = int(pr.sum())

    if gt_sum == 0 and pr_sum == 0:
        return 0.0, 0.0
    if gt_sum == 0 and pr_sum > 0:
        return np.nan, np.nan
    if gt_sum > 0 and pr_sum == 0:
        return np.nan, np.nan

    d_gt_pr = surface_distances(gt, pr, spacing)
    d_pr_gt = surface_distances(pr, gt, spacing)

    if len(d_gt_pr) == 0 or len(d_pr_gt) == 0:
        return np.nan, np.nan

    all_d = np.concatenate([d_gt_pr, d_pr_gt])
    hd95 = float(np.percentile(all_d, 95))
    asd = float(all_d.mean())
    return hd95, asd


def dice_iou(gt: np.ndarray, pr: np.ndarray):
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    g = gt.sum()
    p = pr.sum()
    if g == 0 and p == 0:
        return 1.0, 1.0
    inter = (gt & pr).sum()
    union = (gt | pr).sum()
    dice = (2.0 * inter) / (g + p + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def classify_case(gt_sum, pr_sum, dice):
    if gt_sum == 0 and pr_sum == 0:
        return "TN_empty"
    if gt_sum == 0 and pr_sum > 0:
        return "FP_only"
    if gt_sum > 0 and pr_sum == 0:
        return "FN_only"
    if dice < 0.1:
        return "failure_low_dice"
    return "ok"


def main():
    GT_DIR = first_existing(GT_ROOT_CANDIDATES)
    PRED_DIR = first_existing(PRED_ROOT_CANDIDATES)

    assert GT_DIR is not None, f"GT folder bulunamadı. Adaylar: {GT_ROOT_CANDIDATES}"
    assert PRED_DIR is not None, f"Pred folder bulunamadı. Adaylar: {PRED_ROOT_CANDIDATES}"

    print("GT_DIR  :", GT_DIR)
    print("PRED_DIR:", PRED_DIR)

    gt_files = sorted(list(GT_DIR.glob("*.nii")) + list(GT_DIR.glob("*.nii.gz")))
    pred_files = sorted(list(PRED_DIR.glob("*.nii")) + list(PRED_DIR.glob("*.nii.gz")))

    gt_map = {f.stem.replace(".nii", ""): f for f in gt_files}
    pred_map = {f.stem.replace(".nii", ""): f for f in pred_files}

    common = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    print("GT count   :", len(gt_files))
    print("Pred count :", len(pred_files))
    print("Common     :", len(common))
    assert len(common) > 0, "GT ve Pred dosya isimleri kesişmiyor."

    rows = []
    for k in common:
        gt_arr, gt_spacing = load_nii(gt_map[k])
        pr_arr, _ = load_nii(pred_map[k])

        gt_bin = to_binary(gt_arr)
        pr_bin = to_binary(pr_arr)

        spacing = gt_spacing
        if len(spacing) < gt_bin.ndim:
            spacing = tuple([1.0] * gt_bin.ndim)

        dsc, iou = dice_iou(gt_bin, pr_bin)
        gt_sum = int(gt_bin.sum())
        pr_sum = int(pr_bin.sum())

        hd95, asd = hd95_asd(gt_bin, pr_bin, spacing)
        cat = classify_case(gt_sum, pr_sum, dsc)

        rows.append({
            "case": k,
            "dice": dsc,
            "iou": iou,
            "A_gt_px2": gt_sum,
            "A_pred_px2": pr_sum,
            "HD95": hd95,
            "ASD": asd,
            "category": cat
        })

    df = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    out_csv = Path("/kaggle/working/boundary_metrics_test50.csv")
    df.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    total = len(df)
    fail_low = int((df["category"] == "failure_low_dice").sum())
    fp_only = int((df["category"] == "FP_only").sum())
    fn_only = int((df["category"] == "FN_only").sum())
    tn_empty = int((df["category"] == "TN_empty").sum())
    ok = int((df["category"] == "ok").sum())

    print("\n=== COUNTS ===")
    print("Total:", total)
    print("OK:", ok)
    print("TN_empty:", tn_empty)
    print("FP_only:", fp_only)
    print("FN_only:", fn_only)
    print("failure_low_dice:", fail_low)

    failure_rate = (fail_low + fp_only + fn_only) / max(total, 1)
    print("\nFailure rate (FP_only + FN_only + low_dice):", failure_rate)

    summary = df[["dice", "iou", "HD95", "ASD"]].agg(["mean", "std", "median"])
    print("\n=== SUMMARY ===")
    print(summary)

    pos = df[df["A_gt_px2"] > 0].copy()
    summary_pos = pos[["dice", "iou", "HD95", "ASD"]].agg(["mean", "std", "median"])
    print("\n=== SUMMARY (POSITIVE GT ONLY) ===")
    print(summary_pos)


if __name__ == "__main__":
    main()
