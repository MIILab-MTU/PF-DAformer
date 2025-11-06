import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# === Configurations ===
PREDICTIONS_CSV = "/workspace/QCT_Segmentation/pfdaformer/runs/run_2/all_cases.csv"
GROUND_TRUTH_CSV = "/workspace/QCT_Segmentation/pfdaformer/data/tulane_qct/validation.csv"
PREDICTIONS_DIR = "/workspace/QCT_Segmentation/pfdaformer/runs/run_2/"
OUTPUT_CSV = "/workspace/QCT_Segmentation/pfdaformer/visualize/radiomics_correlation_summary.csv"

# Derived paths
OUTPUT_DIR = os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helper utilities ----------
def feature_suffix(feat_key: str) -> str:
    """Make a unique, readable suffix for column names (strip 'original_' only)."""
    return feat_key.replace("original_", "")

def collect_feature_pairs(columns):
    """Find GT_/Pred_ feature column pairs by suffix; robust to either old or new naming."""
    gt_cols = [c for c in columns if c.startswith("GT_")]
    pred_cols = [c for c in columns if c.startswith("Pred_")]
    gt_suffixes = {c[3:]: c for c in gt_cols}      # suffix -> full col name
    pred_suffixes = {c[5:]: c for c in pred_cols}
    common_suffixes = sorted(set(gt_suffixes.keys()) & set(pred_suffixes.keys()))
    return [(suf, gt_suffixes[suf], pred_suffixes[suf]) for suf in common_suffixes]

def plot_per_feature_line(corr_df: pd.DataFrame, out_png: str, top_n: int = 40):
    # Limit to top_n for readability
    data = corr_df.head(top_n).reset_index(drop=True)

    plt.figure(figsize=(max(8, 0.25 * len(data)), 4.5))
    plt.plot(range(len(data)), data["pearson_r"], marker="o", linestyle="-", linewidth=1.5)
    plt.xticks(range(len(data)), data["feature"], rotation=45, ha="right")
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Mean Pearson r (across cases)")
    plt.title("Per-Feature Agreement (GT vs Pred)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def compute_per_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for suf, gt_col, pr_col in collect_feature_pairs(df.columns):
        x = pd.to_numeric(df[gt_col], errors='coerce')
        y = pd.to_numeric(df[pr_col], errors='coerce')
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 2:
            r = np.corrcoef(x[valid], y[valid])[0, 1]
        else:
            r = np.nan
        rows.append({"feature": suf, "pearson_r": r, "n": int(valid.sum())})
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)

def plot_histogram_of_case_corr(df: pd.DataFrame, out_png: str):
    vals = pd.to_numeric(df["feature_correlation"], errors='coerce').dropna()
    plt.figure()
    plt.hist(vals, bins=20, edgecolor='k')
    plt.xlabel("Feature Correlation (GT vs Pred)")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Radiomics Feature Correlations")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_per_feature_bar(corr_df: pd.DataFrame, out_png: str, top_n: int = 40):
    # Limit to top_n for readability
    data = corr_df.head(top_n)
    plt.figure(figsize=(max(8, 0.22 * len(data)), 4.5))
    plt.bar(data["feature"], data["pearson_r"])
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Mean Pearson r (across cases)")
    plt.title("Per-Feature Agreement (GT vs Pred)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_scatter_for_feature(df: pd.DataFrame, suffix: str, out_png: str):
    # find matching GT_/Pred_ columns by suffix
    pairs = collect_feature_pairs(df.columns)
    match = [p for p in pairs if p[0] == suffix]
    if not match:
        return
    _, gt_col, pr_col = match[0]
    x = pd.to_numeric(df[gt_col], errors='coerce')
    y = pd.to_numeric(df[pr_col], errors='coerce')
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return
    xv, yv = x[valid], y[valid]
    lim_min = min(xv.min(), yv.min())
    lim_max = max(xv.max(), yv.max())
    plt.figure()
    plt.scatter(xv, yv, alpha=0.6)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle='--')
    plt.xlabel(f"GT {suffix}")
    plt.ylabel(f"Pred {suffix}")
    plt.title(f"GT vs Pred for {suffix}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def generate_plots(df: pd.DataFrame):
    # 1) Histogram of per-case vector correlation
    plot_histogram_of_case_corr(df, os.path.join(OUTPUT_DIR, "feature_corr_hist.png"))

    # 2) Per-feature correlation across cases (line plot)
    per_feat = compute_per_feature_correlations(df)
    per_feat.to_csv(os.path.join(OUTPUT_DIR, "radiomics_per_feature_correlation.csv"), index=False)
    plot_per_feature_line(per_feat, os.path.join(OUTPUT_DIR, "per_feature_corr_line.png"))


    # 2) Per-feature correlation across cases (bar)
    per_feat = compute_per_feature_correlations(df)
    per_feat.to_csv(os.path.join(OUTPUT_DIR, "radiomics_per_feature_correlation.csv"), index=False)
    plot_per_feature_bar(per_feat, os.path.join(OUTPUT_DIR, "per_feature_corr_bar.png"))

    # 3) A few representative scatter plots (tweak list as you like)
    # Use suffixes that exist in your CSV (after 'GT_'/'Pred_'), e.g., 'firstorder_Mean', 'glcm_Contrast'
    candidate_suffixes = [
        "firstorder_Mean",
        "glcm_Contrast",
        "glrlm_ShortRunEmphasis",
        "ngtdm_Coarseness"
    ]
    for suf in candidate_suffixes:
        out_png = os.path.join(OUTPUT_DIR, f"scatter_{suf.replace('/','-')}.png")
        plot_scatter_for_feature(df, suf, out_png)

    # Summary stats table for Discussion
    case_corr = pd.to_numeric(df["feature_correlation"], errors='coerce')
    summary = pd.DataFrame({
        "n_cases": [int(case_corr.notna().sum())],
        "mean_case_corr": [float(case_corr.mean(skipna=True))],
        "std_case_corr": [float(case_corr.std(skipna=True))],
        "median_case_corr": [float(case_corr.median(skipna=True))],
        "p80_case_corr": [float(case_corr.quantile(0.8))],
        "p90_case_corr": [float(case_corr.quantile(0.9))]
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "radiomics_case_corr_summary.csv"), index=False)
    print("Saved plots and summaries to:", OUTPUT_DIR)

# ---------- Main ----------
def run_full_pipeline_and_save():
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    matched_cases = pred_df[pred_df["case_name"].isin(gt_df["case_name"])]

    # Configure PyRadiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()


    # extractor.disableAllImageTypes()
    # extractor.enableImageTypeByName('Original')

    # Curated radiomics set for QCT
    feature_names = [
        # First-order
        'original_firstorder_Mean',
        'original_firstorder_Median',
        'original_firstorder_Variance',
        'original_firstorder_Skewness',
        'original_firstorder_Kurtosis',
        'original_firstorder_Entropy',
        'original_firstorder_Energy',
        'original_firstorder_10Percentile',
        'original_firstorder_90Percentile',
        'original_firstorder_InterquartileRange',
        'original_firstorder_MeanAbsoluteDeviation',
        # GLCM
        'original_glcm_Contrast',
        'original_glcm_Correlation',
        'original_glcm_JointEntropy',
        'original_glcm_JointEnergy',
        'original_glcm_Autocorrelation',
        'original_glcm_Id',
        'original_glcm_Idm',
        'original_glcm_InverseVariance',
        'original_glcm_DifferenceEntropy',
        'original_glcm_SumSquares',
        # GLRLM
        'original_glrlm_ShortRunEmphasis',
        'original_glrlm_LongRunEmphasis',
        'original_glrlm_GrayLevelNonUniformity',
        'original_glrlm_RunLengthNonUniformity',
        'original_glrlm_RunPercentage',
        'original_glrlm_HighGrayLevelRunEmphasis',
        'original_glrlm_LowGrayLevelRunEmphasis',
        # GLSZM
        'original_glszm_SmallAreaEmphasis',
        'original_glszm_LargeAreaEmphasis',
        'original_glszm_GrayLevelNonUniformity',
        'original_glszm_ZonePercentage',
        'original_glszm_HighGrayLevelZoneEmphasis',
        'original_glszm_LowGrayLevelZoneEmphasis',
        # NGTDM
        'original_ngtdm_Coarseness',
        'original_ngtdm_Contrast',
        'original_ngtdm_Busyness',
        'original_ngtdm_Strength',
        'original_ngtdm_Complexity',
        # Shape
        'original_shape_VoxelVolume',
        'original_shape_SurfaceArea',
        'original_shape_SurfaceVolumeRatio',
        'original_shape_Sphericity',
    ]

    all_results = []

    for _, row in tqdm(matched_cases.iterrows(), total=len(matched_cases), desc="PyRadiomics + Correlation"):
        case_id = row["case_name"]
        pred_path = os.path.join(PREDICTIONS_DIR, case_id, f"{case_id}_label.pt")
        gt_row = gt_df[gt_df["case_name"] == case_id].iloc[0]
        gt_path = os.path.join(gt_row["data_path"], f"{case_id}_label.pt")
        o_image_path = os.path.join(gt_row["data_path"], f"{case_id}_modalities.pt")

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"[warning] Missing file for {case_id}")
            continue

        # Load volumes (binary masks)
        pred_np = torch.load(pred_path).squeeze().cpu().numpy().astype(np.uint8)
        gt_np = torch.load(gt_path).squeeze().cpu().numpy().astype(np.uint8)
        orig_image_np = torch.load(o_image_path).squeeze().cpu().numpy().astype(np.uint8)

        if pred_np.shape != gt_np.shape:
            print(f"[warning] Shape mismatch for {case_id}")
            continue

        # IMPORTANT: For valid intensity features, 'image' should be the actual CT HU volume.
        # This line uses the mask as the image (placeholder). Replace with HU volume load in your pipeline.
        image = sitk.GetImageFromArray(orig_image_np.astype(np.float32))  # <-- replace with CT
        gt_mask = sitk.GetImageFromArray(gt_np)
        pred_mask = sitk.GetImageFromArray(pred_np)

        try:
            gt_feats = extractor.execute(image, gt_mask)
            pred_feats = extractor.execute(image, pred_mask)

            # Vector-level correlation across selected features
            gt_values = [gt_feats.get(f, np.nan) for f in feature_names]
            pred_values = [pred_feats.get(f, np.nan) for f in feature_names]
            if all(np.isfinite(gt_values)) and all(np.isfinite(pred_values)):
                corr, _ = pearsonr(gt_values, pred_values)
            else:
                corr = np.nan

            # Use unambiguous column names
            result = {"case_id": case_id, "feature_correlation": corr}
            for f in feature_names:
                suf = feature_suffix(f)  # e.g., 'glcm_Contrast'
                result[f"GT_{suf}"] = gt_feats.get(f, np.nan)
                result[f"Pred_{suf}"] = pred_feats.get(f, np.nan)

            all_results.append(result)

        except Exception as e:
            print(f"[error] Failed on {case_id}: {e}")
            continue

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTexture features + correlation saved to: {OUTPUT_CSV}")
    return df

def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"[info] Found existing summary CSV at: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        print("[info] Summary CSV not found; running full extraction...")
        df = run_full_pipeline_and_save()

    # Generate plots and summaries
    if "feature_correlation" not in df.columns:
        raise ValueError("Expected 'feature_correlation' column not found in summary CSV.")
    generate_plots(df)
    print("Done.")

if __name__ == "__main__":
    main()
