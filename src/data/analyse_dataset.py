#!/usr/bin/env python3
"""
analyse_dataset.py

Analyzes H5 files containing antibody-antigen predictions, generating visualizations
and statistics about the data distribution, feature correlations, and class imbalances.
"""
import os
import argparse
import pandas as pd
import numpy as np
import h5py
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Global DockQ bin edges (keeping the same binning logic)
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.00]
NUM_BUCKETS = len(BIN_EDGES) - 1
def load_data(h5_dir):
    """Load data from H5 files into a pandas DataFrame (robust to missing keys)."""
    def first_existing(hf, base, keys):
        """Return (key, dataset) for the first key under base that exists, else (None, None)."""
        for k in keys:
            path = f"{base}/{k}"
            if path in hf:
                return k, hf[path]
        return None, None

    def read_scalar(ds):
        """Read scalar/0-D dataset to Python float safely."""
        try:
            v = ds[()]
            if isinstance(v, np.ndarray):
                # 0-D array -> scalar
                v = v.item() if v.shape == () else float(np.asarray(v).squeeze()[0])
            return float(v)
        except Exception:
            return np.nan

    rows = []
    pae_stats_rows = []

    for fname in sorted(os.listdir(h5_dir)):
        if not fname.endswith('.h5'):
            continue
        h5_path = os.path.join(h5_dir, fname)
        try:
            with h5py.File(h5_path, 'r') as hf:
                for complex_id in hf.keys():
                    grp = hf[complex_id]
                    for sample in grp.keys():
                        base = f"{complex_id}/{sample}"

                        # --- robust key lookups ---
                        _, dockq_ds = first_existing(hf, base, ["abag_dockq", "dockq"])
                        _, ptm_ds   = first_existing(hf, base, ["ptm"])
                        _, iptm_ds  = first_existing(hf, base, ["iptm"])
                        _, rc_ds    = first_existing(hf, base, ["ranking_confidence", "ranking_confidence_af"])
                        _, tmn_ds   = first_existing(hf, base, ["tm_normalized_reference", "tm_normalized"])

                        # require these to exist
                        if dockq_ds is None or ptm_ds is None or iptm_ds is None or rc_ds is None or tmn_ds is None:
                            # skip this sample if anything critical is missing
                            # (optional) print a short note:
                            # print(f"[skip missing] {h5_path}:{base}")
                            continue

                        dockq = read_scalar(dockq_ds)
                        ptm = read_scalar(ptm_ds)
                        iptm = read_scalar(iptm_ds)
                        ranking_confidence = read_scalar(rc_ds)
                        tm_normalized = read_scalar(tmn_ds)

                        # sanity checks
                        if any(np.isnan(x) for x in [dockq, ptm, iptm, ranking_confidence, tm_normalized]):
                            continue
                        if not (0.0 <= dockq <= 1.0):
                            continue

                        bucket = int(np.digitize(dockq, BIN_EDGES, right=False) - 1)
                        bucket = max(0, min(NUM_BUCKETS - 1, bucket))

                        # PAE stats are optional; skip if absent
                        pae_path = f"{base}/interchain_pae_vals"
                        pae_vals = None
                        if pae_path in hf:
                            try:
                                pae_vals = np.asarray(hf[pae_path][()], dtype=float).ravel()
                                if pae_vals.size == 0 or np.any(~np.isfinite(pae_vals)):
                                    pae_vals = None
                            except Exception:
                                pae_vals = None

                        if pae_vals is not None:
                            pae_stats_rows.append({
                                'complex_id': complex_id,
                                'sample': sample,
                                'pae_mean': float(np.mean(pae_vals)),
                                'pae_std': float(np.std(pae_vals)),
                                'pae_median': float(np.median(pae_vals)),
                                'pae_min': float(np.min(pae_vals)),
                                'pae_max': float(np.max(pae_vals)),
                                'pae_q25': float(np.percentile(pae_vals, 25)),
                                'pae_q75': float(np.percentile(pae_vals, 75)),
                                'dockq': dockq,
                                'tm_normalized': tm_normalized
                            })

                        rows.append({
                            'complex_id': complex_id,
                            'sample': sample,
                            'dockq': dockq,
                            'ptm': ptm,
                            'iptm': iptm,
                            'ranking_confidence': ranking_confidence,
                            'tm_normalized': tm_normalized,
                            'bucket': bucket
                        })
        except (OSError, KeyError) as e:
            # Bad/corrupted file or unreadable object table: skip file and keep going
            print(f"[warn] Skipping file due to error: {h5_path} ({type(e).__name__}: {e})")
            continue

    df = pd.DataFrame(rows)
    pae_df = pd.DataFrame(pae_stats_rows)
    return df, pae_df


def plot_feature_distributions(df):
    """Create violin plots for main features."""
    features = ['dockq', 'ptm', 'iptm', 'ranking_confidence', 'tm_normalized']
    
    plt.figure(figsize=(15, 10))
    
    # Overall distributions
    plt.subplot(2, 1, 1)
    sns.violinplot(data=df[features])
    plt.title('Overall Feature Distributions')
    plt.xticks(rotation=45)
    
    # Distributions by bucket
    plt.subplot(2, 1, 2)
    df_melted = df.melt(id_vars=['bucket'], value_vars=features, 
                        var_name='Feature', value_name='Value')
    sns.violinplot(data=df_melted, x='Feature', y='Value', hue='bucket')
    plt.title('Feature Distributions by DockQ Bucket')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_output/feature_distributions.png')
    plt.close()

def analyze_class_imbalance(df):
    """Analyze and visualize class imbalance."""
    bucket_counts = df['bucket'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    bucket_counts.plot(kind='bar')
    plt.title('Distribution of Samples Across DockQ Buckets')
    plt.xlabel('Bucket')
    plt.ylabel('Count')
    
    # Calculate imbalance ratios
    minority_count = bucket_counts.min()
    majority_count = bucket_counts.max()
    imbalance_ratio = minority_count / majority_count
    
    print(f"\nClass Imbalance Analysis:")
    print(f"Minority to Majority Ratio: {imbalance_ratio:.3f}")
    print("\nBucket Frequencies:")
    for bucket, count in bucket_counts.items():
        print(f"Bucket {bucket}: {count} samples ({count/len(df)*100:.1f}%)")
    
    plt.savefig('analysis_output/class_imbalance.png')
    plt.close()

def analyze_feature_statistics(df):
    """Compute and display feature statistics."""
    features = ['dockq', 'ptm', 'iptm', 'ranking_confidence']
    
    stats_dict = {
        'mean': df[features].mean(),
        'std': df[features].std(),
        'skew': df[features].skew(),
        'kurtosis': df[features].kurtosis()
    }
    
    stats_df = pd.DataFrame(stats_dict)
    print("\nFeature Statistics:")
    print(stats_df)
    
    return stats_df

def analyze_pae_correlations(pae_df):
    """Analyze correlations between PAE statistics and quality metrics."""
    # Select features for correlation analysis
    pae_features = ['pae_mean', 'pae_std', 'pae_median', 'pae_min', 'pae_max', 'pae_q25', 'pae_q75']
    quality_metrics = ['dockq', 'tm_normalized']
    
    # Compute correlation matrix
    corr_matrix = pae_df[pae_features + quality_metrics].corr()
    
    # Plot correlation heatmap for PAE features vs quality metrics
    plt.figure(figsize=(12, 8))
    # Select only the correlations between PAE features and quality metrics
    corr_subset = corr_matrix.loc[pae_features, quality_metrics]
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('PAE Features vs Quality Metrics Correlations')
    plt.tight_layout()
    plt.savefig('analysis_output/pae_correlations.png')
    plt.close()
    
    # Print strongest correlations
    print("\nStrongest PAE Correlations with Quality Metrics:")
    for metric in quality_metrics:
        print(f"\nCorrelations with {metric}:")
        correlations = corr_subset[metric].sort_values(ascending=False)
        for feat, corr in correlations.items():
            print(f"{feat}: {corr:.3f}")

def analyze_feature_correlations(df):
    """Analyze and visualize feature correlations."""
    features = ['dockq', 'ptm', 'iptm', 'ranking_confidence', 'tm_normalized']
    
    # Compute correlation matrix
    corr_matrix = df[features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlations')
    plt.savefig('analysis_output/correlations.png')
    plt.close()
    
    # Find highly correlated features
    print("\nHighly Correlated Features (|r| > 0.9):")
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i,j]) > 0.9:
                print(f"{features[i]} - {features[j]}: {corr_matrix.iloc[i,j]:.3f}")

def compute_vif(df):
    """Compute Variance Inflation Factor for features."""
    features = ['ptm', 'iptm', 'ranking_confidence']  # excluding dockq as it's our target
    X = df[features]
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVariance Inflation Factors:")
    print(vif_data)

def main():
    parser = argparse.ArgumentParser(description='Analyze antibody-antigen prediction dataset')
    parser.add_argument('--h5_dir', default='/proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/new/abag_dataset_new_with_stats_filtered/', help='Directory containing H5 files')
    args = parser.parse_args()
    
    # Create output directory for plots
    os.makedirs('analysis_output', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df, pae_df = load_data(args.h5_dir)
    print(f"Loaded {len(df)} samples from {df['complex_id'].nunique()} complexes")
    
    # Generate all analyses
    print("\nGenerating visualizations and analyses...")
    plot_feature_distributions(df)
    analyze_class_imbalance(df)
    stats_df = analyze_feature_statistics(df)
    analyze_feature_correlations(df)
    analyze_pae_correlations(pae_df)
    compute_vif(df)
    
    # Save statistics to CSV
    stats_df.to_csv('analysis_output/feature_statistics.csv')
    pae_df.to_csv('analysis_output/pae_statistics.csv')
    
    print("\nAnalysis complete. Check the analysis_output directory for results.")

if __name__ == '__main__':
    main()
