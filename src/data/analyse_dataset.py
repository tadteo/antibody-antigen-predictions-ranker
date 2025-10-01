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
    """Load data from H5 files into a pandas DataFrame (handles new structure with complex-level stats)."""
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
                    
                    # Load complex-level PAE statistics if available
                    complex_pae_stats = {}
                    pae_stat_keys = ['pae_col_mean', 'pae_col_median', 'pae_col_std']
                    for stat_key in pae_stat_keys:
                        if stat_key in grp:
                            try:
                                complex_pae_stats[stat_key] = read_scalar(grp[stat_key])
                            except Exception:
                                complex_pae_stats[stat_key] = np.nan
                    
                    # Load complex-level CA distance statistics if available
                    complex_ca_stats = {}
                    ca_stat_keys = ['ca_distances_col_mean', 'ca_distances_col_median', 'ca_distances_col_std']
                    for stat_key in ca_stat_keys:
                        if stat_key in grp:
                            try:
                                complex_ca_stats[stat_key] = read_scalar(grp[stat_key])
                            except Exception:
                                complex_ca_stats[stat_key] = np.nan
                    
                    # Iterate through samples (looking for seed-* pattern or other sample names)
                    for sample in grp.keys():
                        # Skip complex-level statistics, only process samples
                        if sample in pae_stat_keys + ca_stat_keys:
                            continue
                            
                        base = f"{complex_id}/{sample}"
                        sample_grp = grp[sample]
                        
                        # Check if this is actually a sample group (has the expected structure)
                        if not isinstance(sample_grp, h5py.Group):
                            continue

                        # --- robust key lookups for sample-level data ---
                        _, dockq_ds = first_existing(hf, base, ["abag_dockq", "dockq"])
                        _, ptm_ds   = first_existing(hf, base, ["ptm"])
                        _, iptm_ds  = first_existing(hf, base, ["iptm"])
                        _, rc_ds    = first_existing(hf, base, ["ranking_score", "ranking_score_af"])
                        _, tmn_ds   = first_existing(hf, base, ["tm_normalized_reference", "tm_normalized"])
                        
                        # For this new structure, some metrics might not be present - let's be more flexible
                        if dockq_ds is None:
                            print(f"[skip missing dockq] {h5_path}:{base}")
                            continue

                        dockq = read_scalar(dockq_ds)
                        ptm = read_scalar(ptm_ds) if ptm_ds is not None else np.nan
                        iptm = read_scalar(iptm_ds) if iptm_ds is not None else np.nan
                        ranking_score = read_scalar(rc_ds) if rc_ds is not None else np.nan
                        tm_normalized = read_scalar(tmn_ds) if tmn_ds is not None else np.nan

                        # sanity checks - only require dockq to be valid
                        if np.isnan(dockq) or not (0.0 <= dockq <= 1.0):
                            continue

                        bucket = int(np.digitize(dockq, BIN_EDGES, right=False) - 1)
                        bucket = max(0, min(NUM_BUCKETS - 1, bucket))

                        # Load sample-level PAE values if available
                        pae_path = f"{base}/interchain_pae_vals"
                        pae_vals = None
                        if pae_path in hf:
                            try:
                                pae_vals = np.asarray(hf[pae_path][()], dtype=float).ravel()
                                if pae_vals.size == 0 or np.any(~np.isfinite(pae_vals)):
                                    pae_vals = None
                            except Exception:
                                pae_vals = None

                        # Create PAE stats row (combining sample-level and complex-level stats)
                        if pae_vals is not None or complex_pae_stats:
                            pae_row = {
                                'complex_id': complex_id,
                                'sample': sample,
                                'dockq': dockq,
                                'tm_normalized': tm_normalized
                            }
                            
                            # Add sample-level PAE stats if available
                            if pae_vals is not None:
                                pae_row.update({
                                    'pae_mean': float(np.mean(pae_vals)),
                                    'pae_std': float(np.std(pae_vals)),
                                    'pae_median': float(np.median(pae_vals)),
                                    'pae_min': float(np.min(pae_vals)),
                                    'pae_max': float(np.max(pae_vals)),
                                    'pae_q25': float(np.percentile(pae_vals, 25)),
                                    'pae_q75': float(np.percentile(pae_vals, 75))
                                })
                            
                            # Add complex-level PAE stats
                            pae_row.update(complex_pae_stats)
                            # Add complex-level CA distance stats
                            pae_row.update(complex_ca_stats)
                            
                            pae_stats_rows.append(pae_row)

                        # Create main data row
                        row = {
                            'complex_id': complex_id,
                            'sample': sample,
                            'dockq': dockq,
                            'ptm': ptm,
                            'iptm': iptm,
                            'ranking_score': ranking_score,
                            'tm_normalized': tm_normalized,
                            'bucket': bucket
                        }
                        
                        # Add complex-level stats to main data as well
                        row.update(complex_pae_stats)
                        row.update(complex_ca_stats)
                        
                        rows.append(row)
                        
        except (OSError, KeyError) as e:
            # Bad/corrupted file or unreadable object table: skip file and keep going
            print(f"[warn] Skipping file due to error: {h5_path} ({type(e).__name__}: {e})")
            continue

    df = pd.DataFrame(rows)
    pae_df = pd.DataFrame(pae_stats_rows)
    return df, pae_df


def plot_feature_distributions(df):
    """Create violin plots for main features."""
    # Core features that should always be present
    core_features = ['dockq']
    
    # Optional features - only include if they exist and have non-NaN values
    optional_features = ['ptm', 'iptm', 'ranking_score', 'tm_normalized']
    available_features = []
    
    for feat in optional_features:
        if feat in df.columns and not df[feat].isna().all():
            available_features.append(feat)
    
    features = core_features + available_features
    
    # Also include complex-level statistics if available
    complex_features = []
    for feat in ['pae_col_mean', 'pae_col_median', 'pae_col_std', 
                 'ca_distances_col_mean', 'ca_distances_col_median', 'ca_distances_col_std']:
        if feat in df.columns and not df[feat].isna().all():
            complex_features.append(feat)
    
    # Create separate plots for different feature groups
    if features:
        plt.figure(figsize=(15, 10))
        
        # Overall distributions for core features
        plt.subplot(2, 1, 1)
        sns.violinplot(data=df[features])
        plt.title('Core Feature Distributions')
        plt.xticks(rotation=45)
        
        # Distributions by bucket
        plt.subplot(2, 1, 2)
        df_melted = df.melt(id_vars=['bucket'], value_vars=features, 
                            var_name='Feature', value_name='Value')
        sns.violinplot(data=df_melted, x='Feature', y='Value', hue='bucket')
        plt.title('Core Feature Distributions by DockQ Bucket')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_output/feature_distributions.png')
        plt.close()
    
    # Plot complex-level features separately if available
    if complex_features:
        plt.figure(figsize=(15, 8))
        
        # Overall distributions for complex features
        plt.subplot(2, 1, 1)
        sns.violinplot(data=df[complex_features])
        plt.title('Complex-Level Feature Distributions')
        plt.xticks(rotation=45)
        
        # Distributions by bucket
        plt.subplot(2, 1, 2)
        df_melted = df.melt(id_vars=['bucket'], value_vars=complex_features, 
                            var_name='Feature', value_name='Value')
        sns.violinplot(data=df_melted, x='Feature', y='Value', hue='bucket')
        plt.title('Complex-Level Feature Distributions by DockQ Bucket')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_output/complex_feature_distributions.png')
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
    # Find available features dynamically
    potential_features = ['dockq', 'ptm', 'iptm', 'ranking_score', 'tm_normalized',
                         'pae_col_mean', 'pae_col_median', 'pae_col_std',
                         'ca_distances_col_mean', 'ca_distances_col_median', 'ca_distances_col_std']
    
    available_features = []
    for feat in potential_features:
        if feat in df.columns and not df[feat].isna().all():
            available_features.append(feat)
    
    if not available_features:
        print("No numeric features available for statistics")
        return pd.DataFrame()
    
    stats_dict = {
        'mean': df[available_features].mean(),
        'std': df[available_features].std(),
        'skew': df[available_features].skew(),
        'kurtosis': df[available_features].kurtosis(),
        'count': df[available_features].count(),
        'missing_pct': (df[available_features].isna().sum() / len(df) * 100)
    }
    
    stats_df = pd.DataFrame(stats_dict)
    print("\nFeature Statistics:")
    print(stats_df.round(3))
    
    return stats_df

def analyze_pae_correlations(pae_df):
    """Analyze correlations between PAE statistics and quality metrics."""
    if pae_df.empty:
        print("No PAE data available for correlation analysis")
        return
    
    # Find available PAE features dynamically
    all_pae_features = ['pae_mean', 'pae_std', 'pae_median', 'pae_min', 'pae_max', 'pae_q25', 'pae_q75',
                       'pae_col_mean', 'pae_col_median', 'pae_col_std',
                       'ca_distances_col_mean', 'ca_distances_col_median', 'ca_distances_col_std']
    
    available_pae_features = []
    for feat in all_pae_features:
        if feat in pae_df.columns and not pae_df[feat].isna().all():
            available_pae_features.append(feat)
    
    quality_metrics = []
    for metric in ['dockq', 'tm_normalized']:
        if metric in pae_df.columns and not pae_df[metric].isna().all():
            quality_metrics.append(metric)
    
    if not available_pae_features or not quality_metrics:
        print("Insufficient data for PAE correlation analysis")
        return
    
    # Compute correlation matrix
    all_features = available_pae_features + quality_metrics
    corr_matrix = pae_df[all_features].corr()
    
    # Plot correlation heatmap for PAE features vs quality metrics
    plt.figure(figsize=(12, 8))
    # Select only the correlations between PAE features and quality metrics
    corr_subset = corr_matrix.loc[available_pae_features, quality_metrics]
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('PAE Features vs Quality Metrics Correlations')
    plt.tight_layout()
    plt.savefig('analysis_output/pae_correlations.png')
    plt.close()
    
    # Print strongest correlations
    print("\nStrongest PAE Correlations with Quality Metrics:")
    for metric in quality_metrics:
        print(f"\nCorrelations with {metric}:")
        correlations = corr_subset[metric].sort_values(key=abs, ascending=False)
        for feat, corr in correlations.items():
            print(f"{feat}: {corr:.3f}")

def analyze_feature_correlations(df):
    """Analyze and visualize feature correlations."""
    # Find available features dynamically
    potential_features = ['dockq', 'ptm', 'iptm', 'ranking_score', 'tm_normalized',
                         'pae_col_mean', 'pae_col_median', 'pae_col_std',
                         'ca_distances_col_mean', 'ca_distances_col_median', 'ca_distances_col_std']
    
    available_features = []
    for feat in potential_features:
        if feat in df.columns and not df[feat].isna().all():
            available_features.append(feat)
    
    if len(available_features) < 2:
        print("Insufficient features for correlation analysis")
        return
    
    # Compute correlation matrix
    corr_matrix = df[available_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('analysis_output/correlations.png')
    plt.close()
    
    # Find highly correlated features
    print("\nHighly Correlated Features (|r| > 0.9):")
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            if abs(corr_matrix.iloc[i,j]) > 0.9:
                print(f"{available_features[i]} - {available_features[j]}: {corr_matrix.iloc[i,j]:.3f}")
    
    # Print moderately correlated features too
    print("\nModerately Correlated Features (0.7 < |r| <= 0.9):")
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            corr_val = abs(corr_matrix.iloc[i,j])
            if 0.7 < corr_val <= 0.9:
                print(f"{available_features[i]} - {available_features[j]}: {corr_matrix.iloc[i,j]:.3f}")

def compute_vif(df):
    """Compute Variance Inflation Factor for features."""
    # Find available features dynamically (excluding dockq as it's our target)
    potential_features = ['ptm', 'iptm', 'ranking_score', 'tm_normalized',
                         'pae_col_mean', 'pae_col_median', 'pae_col_std',
                         'ca_distances_col_mean', 'ca_distances_col_median', 'ca_distances_col_std']
    
    available_features = []
    for feat in potential_features:
        if feat in df.columns and not df[feat].isna().all():
            available_features.append(feat)
    
    if len(available_features) < 2:
        print("Insufficient features for VIF analysis")
        return
    
    # Remove rows with any NaN values for VIF computation
    X = df[available_features].dropna()
    
    if len(X) == 0:
        print("No complete cases available for VIF analysis")
        return
    
    try:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = available_features
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        print("\nVariance Inflation Factors:")
        print(vif_data.round(3))
    except Exception as e:
        print(f"Error computing VIF: {e}")

def analyze_weight_distribution(manifest_df):
    """Analyze weight distributions across splits and their effect on class balance."""
    if 'weight' not in manifest_df.columns:
        print("No weight column found in manifest")
        return
    
    print("\n" + "="*50)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Overall weight statistics
    print(f"\nOverall Weight Statistics:")
    print(f"Mean: {manifest_df['weight'].mean():.6f}")
    print(f"Std: {manifest_df['weight'].std():.6f}")
    print(f"Min: {manifest_df['weight'].min():.6f}")
    print(f"Max: {manifest_df['weight'].max():.6f}")
    print(f"Sum: {manifest_df['weight'].sum():.2f}")
    
    # Weight statistics by split
    if 'split' in manifest_df.columns:
        print(f"\nWeight Statistics by Split:")
        for split in manifest_df['split'].unique():
            if pd.isna(split):
                continue
            split_data = manifest_df[manifest_df['split'] == split]
            print(f"\n{split.upper()} Split:")
            print(f"  Samples: {len(split_data)}")
            print(f"  Mean weight: {split_data['weight'].mean():.6f}")
            print(f"  Std weight: {split_data['weight'].std():.6f}")
            print(f"  Min weight: {split_data['weight'].min():.6f}")
            print(f"  Max weight: {split_data['weight'].max():.6f}")
            print(f"  Sum of weights: {split_data['weight'].sum():.2f}")
    
    # Create weight distribution visualizations
    plt.figure(figsize=(15, 12))
    
    # 1. Overall weight distribution
    plt.subplot(3, 2, 1)
    plt.hist(manifest_df['weight'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Overall Weight Distribution')
    plt.yscale('log')
    
    # 2. Weight vs DockQ scatter
    plt.subplot(3, 2, 2)
    plt.scatter(manifest_df['label'], manifest_df['weight'], alpha=0.6, s=10)
    plt.xlabel('DockQ')
    plt.ylabel('Weight')
    plt.title('Weight vs DockQ')
    plt.yscale('log')
    
    # 3. Weight distribution by split
    if 'split' in manifest_df.columns:
        plt.subplot(3, 2, 3)
        splits = [s for s in manifest_df['split'].unique() if not pd.isna(s)]
        for split in splits:
            split_weights = manifest_df[manifest_df['split'] == split]['weight']
            plt.hist(split_weights, bins=30, alpha=0.6, label=split, density=True)
        plt.xlabel('Weight')
        plt.ylabel('Density')
        plt.title('Weight Distribution by Split')
        plt.legend()
        plt.yscale('log')
    
    # 4. Effective sample distribution (original vs weighted)
    plt.subplot(3, 2, 4)
    # Create DockQ bins for comparison
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Original distribution
    orig_hist, _ = np.histogram(manifest_df['label'], bins=bins)
    
    # Weighted distribution (approximate)
    weighted_hist = np.zeros_like(orig_hist, dtype=float)
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (manifest_df['label'] >= left) & (manifest_df['label'] < right)
        weighted_hist[i] = manifest_df[mask]['weight'].sum()
    
    plt.plot(bin_centers, orig_hist, 'o-', label='Original', alpha=0.7)
    plt.plot(bin_centers, weighted_hist, 's-', label='Weighted', alpha=0.7)
    plt.xlabel('DockQ')
    plt.ylabel('Count / Weighted Sum')
    plt.title('Original vs Weighted Distribution')
    plt.legend()
    
    # 5. Weight distribution by DockQ bucket
    plt.subplot(3, 2, 5)
    # Use the same bucket logic as in the original code
    manifest_df_temp = manifest_df.copy()
    manifest_df_temp['bucket'] = np.digitize(manifest_df_temp['label'], BIN_EDGES, right=False) - 1
    manifest_df_temp['bucket'] = np.clip(manifest_df_temp['bucket'], 0, NUM_BUCKETS - 1)
    
    bucket_weights = []
    bucket_labels = []
    for bucket in range(NUM_BUCKETS):
        bucket_data = manifest_df_temp[manifest_df_temp['bucket'] == bucket]
        if len(bucket_data) > 0:
            bucket_weights.append(bucket_data['weight'].values)
            bucket_labels.append(f'Bucket {bucket}\n({len(bucket_data)} samples)')
    
    plt.boxplot(bucket_weights, labels=bucket_labels)
    plt.ylabel('Weight')
    plt.title('Weight Distribution by DockQ Bucket')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # 6. Cumulative effect visualization
    plt.subplot(3, 2, 6)
    sorted_indices = np.argsort(manifest_df['label'])
    sorted_labels = manifest_df['label'].iloc[sorted_indices]
    sorted_weights = manifest_df['weight'].iloc[sorted_indices]
    
    # Cumulative original samples
    cum_orig = np.arange(1, len(sorted_labels) + 1)
    # Cumulative weighted samples
    cum_weighted = np.cumsum(sorted_weights)
    
    plt.plot(sorted_labels, cum_orig, label='Original (cumulative count)', alpha=0.7)
    plt.plot(sorted_labels, cum_weighted, label='Weighted (cumulative sum)', alpha=0.7)
    plt.xlabel('DockQ (sorted)')
    plt.ylabel('Cumulative Value')
    plt.title('Cumulative Distribution: Original vs Weighted')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analysis_output/weight_distribution_analysis.png', dpi=150)
    plt.close()
    
    # Calculate and print rebalancing effectiveness
    print(f"\nRebalancing Effectiveness Analysis:")
    
    # Calculate variance reduction
    orig_bucket_counts = manifest_df_temp.groupby('bucket').size()
    weighted_bucket_sums = manifest_df_temp.groupby('bucket')['weight'].sum()
    
    print(f"\nBucket Analysis:")
    print(f"{'Bucket':<8} {'Original':<10} {'Weighted':<12} {'Weight Ratio':<12}")
    print("-" * 45)
    
    for bucket in range(NUM_BUCKETS):
        orig_count = orig_bucket_counts.get(bucket, 0)
        weighted_sum = weighted_bucket_sums.get(bucket, 0.0)
        ratio = weighted_sum / orig_count if orig_count > 0 else 0
        print(f"{bucket:<8} {orig_count:<10} {weighted_sum:<12.3f} {ratio:<12.3f}")
    
    # Calculate coefficient of variation (CV) as a measure of imbalance
    orig_cv = orig_bucket_counts.std() / orig_bucket_counts.mean()
    weighted_cv = weighted_bucket_sums.std() / weighted_bucket_sums.mean()
    
    print(f"\nImbalance Measures (Coefficient of Variation):")
    print(f"Original distribution CV: {orig_cv:.3f}")
    print(f"Weighted distribution CV: {weighted_cv:.3f}")
    print(f"CV reduction: {((orig_cv - weighted_cv) / orig_cv * 100):.1f}%")

def load_manifest_and_analyze_weights(manifest_path):
    """Load manifest CSV and perform weight analysis."""
    try:
        manifest_df = pd.read_csv(manifest_path)
        print(f"Loaded manifest with {len(manifest_df)} samples")
        
        # Check required columns
        required_cols = ['label', 'weight']
        missing_cols = [col for col in required_cols if col not in manifest_df.columns]
        if missing_cols:
            print(f"Missing required columns in manifest: {missing_cols}")
            return None
        
        analyze_weight_distribution(manifest_df)
        return manifest_df
        
    except Exception as e:
        print(f"Error loading manifest: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze antibody-antigen prediction dataset')
    parser.add_argument('--h5_dir', default='/proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/new_test_with_stats_filtered', help='Directory containing H5 files')
    parser.add_argument('--manifest_path', default='/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/data/manifest_new_with_filtered_test_density_with_clipping_500k_maxlen.csv', help='Path to manifest CSV file for weight analysis')
    args = parser.parse_args()
    
    # Create output directory for plots
    os.makedirs('analysis_output', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df, pae_df = load_data(args.h5_dir)
    print(f"Loaded {len(df)} samples from {df['complex_id'].nunique()} complexes")
    print(f"Available columns: {list(df.columns)}")
    print(f"PAE dataframe shape: {pae_df.shape}")
    
    # Generate all analyses
    print("\nGenerating visualizations and analyses...")
    plot_feature_distributions(df)
    analyze_class_imbalance(df)
    stats_df = analyze_feature_statistics(df)
    analyze_feature_correlations(df)
    analyze_pae_correlations(pae_df)
    compute_vif(df)
    
    # Weight analysis if manifest is provided
    manifest_df = None
    if args.manifest_path:
        print(f"\nLoading manifest for weight analysis...")
        manifest_df = load_manifest_and_analyze_weights(args.manifest_path)
    
    # Save statistics to CSV
    if not stats_df.empty:
        stats_df.to_csv('analysis_output/feature_statistics.csv')
    if not pae_df.empty:
        pae_df.to_csv('analysis_output/pae_statistics.csv')
    if manifest_df is not None:
        # Save a summary of weight analysis
        weight_summary = {
            'overall_weight_mean': manifest_df['weight'].mean(),
            'overall_weight_std': manifest_df['weight'].std(),
            'overall_weight_sum': manifest_df['weight'].sum(),
            'total_samples': len(manifest_df)
        }
        
        # Add per-split statistics if available
        if 'split' in manifest_df.columns:
            for split in manifest_df['split'].unique():
                if not pd.isna(split):
                    split_data = manifest_df[manifest_df['split'] == split]
                    weight_summary[f'{split}_weight_mean'] = split_data['weight'].mean()
                    weight_summary[f'{split}_weight_sum'] = split_data['weight'].sum()
                    weight_summary[f'{split}_sample_count'] = len(split_data)
        
        weight_summary_df = pd.DataFrame([weight_summary])
        weight_summary_df.to_csv('analysis_output/weight_analysis_summary.csv', index=False)
    
    print("\nAnalysis complete. Check the analysis_output directory for results.")
    if args.manifest_path:
        print("Weight distribution analysis included - see weight_distribution_analysis.png")
        print("Weight summary saved to weight_analysis_summary.csv")

if __name__ == '__main__':
    main()
