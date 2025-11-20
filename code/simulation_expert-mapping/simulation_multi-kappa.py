"""
============================================================================
MULTI-KAPPA SIMULATION: Validation Across Agreement Levels
Sample Size Determination for Expert Rater Survey

This version runs enhanced analyses across MULTIPLE true kappa values
to validate sample size across different agreement scenarios.

Purpose: Since you don't know true kappa in advance, this lets you:
1. Plan conservatively (use worst-case scenario)
2. Report results conditional on observed kappa
3. Validate adequacy post-hoc after data collection
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
np.random.seed(42)

# ============================================================================
# [INCLUDE ALL PREVIOUS FUNCTIONS: survey_structure, calculate_fleiss_kappa, 
#  simulate_expert_ratings_direct, analyze_stability, etc.]
# For brevity, I'll note where they go but include the NEW/MODIFIED functions
# ============================================================================

# [INSERT: survey_structure definition]
survey_structure = {
    'Depression': {
        'dimensions': ['Mood_Affective', 'Cognitive_SelfPerception', 
                      'Somatic_Vegetative', 'Activity_Interest', 'Anxiety_Distress', 'None'],
        'n_items': 73
    },
    'Anxiety': {
        'dimensions': ['Somatic', 'Cognitive', 'None'],
        'n_items': 83
    },
    'Psychosis': {
        'dimensions': ['Hallucinations', 'Delusions', 'None'],
        'n_items': 37
    },
    'Apathy': {
        'dimensions': ['Cognitive', 'Behavioral', 'Affective', 'None'],
        'n_items': 89
    },
    'ICD': {
        'dimensions': ['Gambling', 'Hypersexuality', 'Buying', 
                      'Eating', 'Punding', 'DDS', 'None'],
        'n_items': 27
    },
    'Sleep': {
        'dimensions': ['Daytime_Sleepiness', 'Nocturnal_Disturbances', 
                      'REM_Behavior', 'Sleep_Breathing', 'None'],
        'n_items': 100
    }
}

# [INSERT: All calculation and simulation functions from previous script]
# calculate_fleiss_kappa_from_ratings, simulate_expert_ratings_direct, etc.
# [Copy from previous script - lines 57-227 from FINAL version]

def calculate_fleiss_kappa_from_ratings(ratings, n_categories):
    """Calculate Fleiss' kappa from raw ratings array."""
    n_items, n_raters = ratings.shape
    P_values = []
    all_ratings = ratings.flatten()
    category_props = np.array([np.mean(all_ratings == k) for k in range(n_categories)])
    P_e = np.sum(category_props ** 2)
    
    for i in range(n_items):
        item_ratings = ratings[i, :]
        counts = np.array([np.sum(item_ratings == k) for k in range(n_categories)])
        sum_squares = np.sum(counts ** 2)
        P_i = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
        P_values.append(P_i)
    
    P_bar = np.mean(P_values)
    
    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

def simulate_expert_ratings_direct(n_raters, n_items, n_categories, target_kappa=0.50):
    """Simulate expert ratings using direct agreement probability method."""
    P_e = 1.0 / n_categories
    required_P = target_kappa * (1 - P_e) + P_e
    agree_prob = np.sqrt(required_P) if required_P >= 0 else 0.0
    agree_prob = np.clip(agree_prob, 1.0/n_categories, 1.0)
    reference_categories = np.random.randint(0, n_categories, size=n_items)
    ratings = np.zeros((n_items, n_raters), dtype=int)
    
    for i in range(n_items):
        ref_cat = reference_categories[i]
        for r in range(n_raters):
            if np.random.random() < agree_prob:
                ratings[i, r] = ref_cat
            else:
                other_cats = [c for c in range(n_categories) if c != ref_cat]
                ratings[i, r] = np.random.choice(other_cats)
    
    return ratings

def analyze_stability(construct_info, true_kappa, max_raters, n_iterations):
    """Analyze stability (from previous script)."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    results = []
    
    for iteration in tqdm(range(n_iterations), desc=f"    κ={true_kappa:.2f} stability", leave=False):
        ratings = simulate_expert_ratings_direct(max_raters, n_items, n_categories, true_kappa)
        kappas_cumulative = []
        
        for n in range(3, max_raters + 1):
            ratings_subset = ratings[:, :n]
            kappa = calculate_fleiss_kappa_from_ratings(ratings_subset, n_categories)
            kappas_cumulative.append(kappa)
        
        for idx, n in enumerate(range(3, max_raters + 1)):
            kappa = kappas_cumulative[idx]
            delta_kappa = abs(kappa - kappas_cumulative[idx - 1]) if idx > 0 else np.nan
            delta_from_final = abs(kappa - kappas_cumulative[-1])
            
            results.append({
                'iteration': iteration,
                'n_raters': n,
                'kappa': kappa,
                'delta_kappa': delta_kappa,
                'delta_from_final': delta_from_final
            })
    
    results_df = pd.DataFrame(results)
    stability_summary = results_df.groupby('n_raters').agg({
        'kappa': ['mean', 'std'],
        'delta_kappa': ['mean', 'std'],
        'delta_from_final': ['mean', 'std']
    }).reset_index()
    
    stability_summary.columns = [
        'n_raters', 'mean_kappa', 'sd_kappa',
        'mean_delta', 'sd_delta',
        'mean_delta_from_final', 'sd_delta_from_final'
    ]
    
    return stability_summary

def analyze_precision_bootstrap(construct_info, n_raters_range, true_kappa, n_iterations, n_bootstrap):
    """Analyze precision with bootstrap (from previous script)."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    results = []
    
    for n_raters in tqdm(n_raters_range, desc=f"    κ={true_kappa:.2f} precision", leave=False):
        ci_widths = []
        
        for iteration in range(n_iterations):
            ratings = simulate_expert_ratings_direct(n_raters, n_items, n_categories, true_kappa)
            kappas_boot = []
            
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n_items, size=n_items, replace=True)
                ratings_boot = ratings[boot_indices, :]
                kappa_boot = calculate_fleiss_kappa_from_ratings(ratings_boot, n_categories)
                kappas_boot.append(kappa_boot)
            
            ci_lower = np.percentile(kappas_boot, 2.5)
            ci_upper = np.percentile(kappas_boot, 97.5)
            ci_widths.append(ci_upper - ci_lower)
        
        results.append({
            'n_raters': n_raters,
            'mean_ci_width': np.mean(ci_widths),
            'sd_ci_width': np.std(ci_widths),
            'median_ci_width': np.median(ci_widths)
        })
    
    return pd.DataFrame(results)

def analyze_replication_variability(construct_info, n_raters_range, true_kappa, n_iterations):
    """Analyze replication variability (from previous script)."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    results = []
    
    for n_raters in tqdm(n_raters_range, desc=f"    κ={true_kappa:.2f} replication", leave=False):
        kappas_all = []
        
        for iteration in range(n_iterations):
            ratings = simulate_expert_ratings_direct(n_raters, n_items, n_categories, true_kappa)
            kappa = calculate_fleiss_kappa_from_ratings(ratings, n_categories)
            kappas_all.append(kappa)
        
        mean_kappa = np.mean(kappas_all)
        sd_across_samples = np.std(kappas_all)
        cv = sd_across_samples / mean_kappa if mean_kappa > 0 else np.nan
        
        percentile_2_5 = np.percentile(kappas_all, 2.5)
        percentile_97_5 = np.percentile(kappas_all, 97.5)
        range_95 = percentile_97_5 - percentile_2_5
        
        percentile_5 = np.percentile(kappas_all, 5)
        percentile_95 = np.percentile(kappas_all, 95)
        range_90 = percentile_95 - percentile_5
        
        results.append({
            'n_raters': n_raters,
            'mean_kappa': mean_kappa,
            'sd_across_samples': sd_across_samples,
            'cv': cv,
            'range_95': range_95,
            'range_90': range_90
        })
    
    return pd.DataFrame(results)

# ============================================================================
# NEW: MULTI-KAPPA COMPARISON PLOTS
# ============================================================================

def plot_stability_comparison(all_stability_df, construct_name, save_path):
    """Compare stability across different true kappa values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'0.40': '#e74c3c', '0.50': '#3498db', '0.60': '#2ecc71', '0.70': '#9b59b6'}
    
    # Plot 1: Mean kappa estimate
    ax = axes[0, 0]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_kappa'],
               marker='o', linewidth=2, markersize=4,
               label=f'True κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean Kappa Estimate')
    ax.set_title('Kappa Estimates by True Agreement Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Change in estimate (stability)
    ax = axes[0, 1]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_delta'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.01, linestyle='--', color='black', alpha=0.5, label='Stable (< 0.01)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean Change in κ')
    ax.set_title('Estimate Stability Across Agreement Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 3: SD (precision)
    ax = axes[1, 0]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['sd_kappa'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('SD of Kappa Estimates')
    ax.set_title('Precision Across Agreement Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 4: Convergence
    ax = axes[1, 1]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_delta_from_final'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.02, linestyle='--', color='black', alpha=0.5, label='Within 0.02')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Distance from Final Estimate')
    ax.set_title('Convergence Across Agreement Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.suptitle(f'Stability Analysis Comparison: {construct_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_precision_comparison(all_precision_df, construct_name, save_path):
    """Compare precision across different true kappa values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'0.40': '#e74c3c', '0.50': '#3498db', '0.60': '#2ecc71', '0.70': '#9b59b6'}
    
    for true_kappa in sorted(all_precision_df['true_kappa'].unique()):
        data = all_precision_df[all_precision_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_ci_width'],
               marker='o', linewidth=2, markersize=6,
               label=f'True κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.10, linestyle='--', color='red', alpha=0.5,
              label='Target (±0.05)')
    ax.axhline(y=0.15, linestyle='--', color='orange', alpha=0.5,
              label='Acceptable (±0.075)')
    
    ax.set_xlabel('Number of Raters', fontsize=12)
    ax.set_ylabel('95% CI Width', fontsize=12)
    ax.set_title(f'Precision Comparison: {construct_name}\n' +
                'Bootstrap CIs Across Agreement Levels',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_replication_comparison(all_replication_df, construct_name, save_path):
    """Compare replication variability across true kappa values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'0.40': '#e74c3c', '0.50': '#3498db', '0.60': '#2ecc71', '0.70': '#9b59b6'}
    
    # SD across samples
    ax = axes[0, 0]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['sd_across_samples'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.05, linestyle='--', color='black', alpha=0.5, label='Target (≤0.05)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('SD Across Samples')
    ax.set_title('Between-Sample Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 95% range
    ax = axes[0, 1]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['range_95'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.10, linestyle='--', color='black', alpha=0.5, label='Target (≤0.10)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('95% Range')
    ax.set_title('Replication Range (95%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 90% range
    ax = axes[1, 0]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['range_90'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.08, linestyle='--', color='black', alpha=0.5, label='Target (≤0.08)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('90% Range')
    ax.set_title('Replication Range (90%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # CV
    ax = axes[1, 1]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['cv'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.10, linestyle='--', color='black', alpha=0.5, label='Target (≤0.10)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Relative Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.suptitle(f'Replication Variability Comparison: {construct_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================================
# NEW: RUN MULTI-KAPPA ANALYSES
# ============================================================================

def run_multi_kappa_analyses(construct_name, true_kappa_range=None):
    """
    Run enhanced analyses across multiple true kappa values.
    
    This allows you to:
    1. Plan conservatively using worst-case (lowest kappa)
    2. Report results conditional on observed kappa
    3. Validate adequacy post-hoc after data collection
    """
    if true_kappa_range is None:
        true_kappa_range = [0.40, 0.50, 0.60, 0.70]
    
    print("\n" + "="*70)
    print(f"MULTI-KAPPA ANALYSES: {construct_name}")
    print(f"Testing κ = {true_kappa_range}")
    print("="*70 + "\n")
    
    construct_info = survey_structure[construct_name]
    output_dir = Path(f'analysis_multikappa_{construct_name.lower()}')
    output_dir.mkdir(exist_ok=True)
    
    all_stability_data = []
    all_precision_data = []
    all_replication_data = []
    all_recommendations = []
    
    # Run analyses for each kappa value
    for true_kappa in true_kappa_range:
        print(f"\n  Analyzing κ = {true_kappa:.2f}...")
        
        # Stability
        stability_df = analyze_stability(
            construct_info=construct_info,
            true_kappa=true_kappa,
            max_raters=30,
            n_iterations=100
        )
        stability_df['true_kappa'] = true_kappa
        all_stability_data.append(stability_df)
        
        # Precision
        precision_df = analyze_precision_bootstrap(
            construct_info=construct_info,
            n_raters_range=[5, 8, 10, 12, 15, 18, 20, 25, 30],
            true_kappa=true_kappa,
            n_iterations=100,
            n_bootstrap=500
        )
        precision_df['true_kappa'] = true_kappa
        all_precision_data.append(precision_df)
        
        # Replication
        replication_df = analyze_replication_variability(
            construct_info=construct_info,
            n_raters_range=[5, 8, 10, 12, 15, 18, 20, 25, 30],
            true_kappa=true_kappa,
            n_iterations=500
        )
        replication_df['true_kappa'] = true_kappa
        all_replication_data.append(replication_df)
        
        # Find thresholds for this kappa
        stable_rows = stability_df[stability_df['mean_delta'] < 0.01]
        stable_n = stable_rows['n_raters'].min() if len(stable_rows) > 0 else 30
        
        precise_rows = precision_df[precision_df['mean_ci_width'] <= 0.10]
        precise_n = precise_rows['n_raters'].min() if len(precise_rows) > 0 else 30
        
        range_rows = replication_df[replication_df['range_95'] <= 0.10]
        range_n = range_rows['n_raters'].min() if len(range_rows) > 0 else 30
        
        overall_n = max(stable_n, precise_n, range_n)
        
        all_recommendations.append({
            'construct': construct_name,
            'true_kappa': true_kappa,
            'stability_n': stable_n,
            'precision_n': precise_n,
            'replication_range_n': range_n,
            'overall_recommendation': overall_n,
            'conservative': overall_n + 3
        })
        
        print(f"    Stability: {stable_n}, Precision: {precise_n}, Range: {range_n} → Rec: {overall_n}")
    
    # Combine all data
    all_stability_df = pd.concat(all_stability_data, ignore_index=True)
    all_precision_df = pd.concat(all_precision_data, ignore_index=True)
    all_replication_df = pd.concat(all_replication_data, ignore_index=True)
    recommendations_df = pd.DataFrame(all_recommendations)
    
    # Save combined data
    all_stability_df.to_csv(output_dir / 'stability_all_kappas.csv', index=False)
    all_precision_df.to_csv(output_dir / 'precision_all_kappas.csv', index=False)
    all_replication_df.to_csv(output_dir / 'replication_all_kappas.csv', index=False)
    recommendations_df.to_csv(output_dir / 'recommendations_by_kappa.csv', index=False)
    
    # Create comparison plots
    print("\n  Creating comparison plots...")
    plot_stability_comparison(all_stability_df, construct_name,
                             save_path=output_dir / 'stability_comparison.png')
    plot_precision_comparison(all_precision_df, construct_name,
                              save_path=output_dir / 'precision_comparison.png')
    plot_replication_comparison(all_replication_df, construct_name,
                               save_path=output_dir / 'replication_comparison.png')
    
    # Print summary
    print("\n" + "="*70)
    print(f"SUMMARY: {construct_name} Across Agreement Levels")
    print("="*70)
    print("\nRecommended sample sizes by true kappa:")
    print(recommendations_df[['true_kappa', 'stability_n', 'precision_n', 
                             'replication_range_n', 'overall_recommendation', 'conservative']])
    
    print(f"\n✓ Results saved to: {output_dir}/")
    
    return recommendations_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run multi-kappa analyses for all constructs."""
    print("="*70)
    print("MULTI-KAPPA SIMULATION STUDY")
    print("Sample Size Validation Across Agreement Levels")
    print("="*70 + "\n")
    
    output_dir = Path('simulation_results_multikappa')
    output_dir.mkdir(exist_ok=True)
    
    all_construct_recommendations = []
    
    for construct_name in survey_structure.keys():
        recommendations_df = run_multi_kappa_analyses(
            construct_name=construct_name,
            true_kappa_range=[0.40, 0.50, 0.60, 0.70]
        )
        all_construct_recommendations.append(recommendations_df)
    
    # Consolidated recommendations across all constructs
    all_recs = pd.concat(all_construct_recommendations, ignore_index=True)
    all_recs.to_csv(output_dir / 'all_constructs_all_kappas.csv', index=False)
    
    print("\n" + "="*70)
    print("COMPLETE! All constructs analyzed across kappa values.")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}/")
    print("\nEach construct has:")
    print("  - stability_all_kappas.csv")
    print("  - precision_all_kappas.csv")
    print("  - replication_all_kappas.csv")
    print("  - recommendations_by_kappa.csv")
    print("  - Comparison plots for each metric")

def quick_test():
    """Quick test on Depression only."""
    print("="*70)
    print("QUICK TEST: Depression Across Multiple Kappas")
    print("="*70 + "\n")
    
    recommendations_df = run_multi_kappa_analyses(
        construct_name='Depression',
        true_kappa_range=[0.40, 0.50, 0.60, 0.70]
    )
    
    print("\n✓ Test complete! Check 'analysis_multikappa_depression/' folder.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        main()
    else:
        print("Running quick test. Use '--full' for all constructs.\n")
        quick_test()