"""
============================================================================
FINAL SIMULATION: Fleiss' Kappa Inter-Rater Reliability
Sample Size Determination for Expert Rater Survey

Includes:
1. Power Analysis: Statistical power across constructs
2. Stability Analysis: When do estimates stabilize?
3. Precision Analysis: Bootstrap confidence intervals
4. Replication Analysis: Variability across independent samples

CLEANED VERSION: Removed invalid ICC/correlation metrics
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
# SURVEY STRUCTURE
# ============================================================================

survey_structure = {
    'Depression': {
        'dimensions': ['Mood_Affective', 'Cognitive_SelfPerception', 
                      'Somatic_Vegetative', 'Activity_Interest', 'Anxiety_Distress'],
        'n_items': 73
    },
    'Anxiety': {
        'dimensions': ['Somatic', 'Cognitive'],
        'n_items': 83
    },
    'Psychosis': {
        'dimensions': ['Hallucinations', 'Delusions'],
        'n_items': 37
    },
    'Apathy': {
        'dimensions': ['Cognitive', 'Behavioral', 'Affective'],
        'n_items': 89
    },
    'ICD': {
        'dimensions': ['Gambling', 'Hypersexuality', 'Buying', 
                      'Eating', 'Punding', 'DDS'],
        'n_items': 27
    },
    'Sleep': {
        'dimensions': ['Daytime_Sleepiness', 'Nocturnal_Disturbances', 
                      'REM_Behavior', 'Sleep_Breathing'],
        'n_items': 100
    }
}

# ============================================================================
# FLEISS' KAPPA CALCULATION
# ============================================================================

def calculate_fleiss_kappa_from_ratings(ratings, n_categories):
    """Calculate Fleiss' kappa from raw ratings array."""
    n_items, n_raters = ratings.shape
    
    # Accumulate across all items
    P_values = []
    
    # Calculate proportion in each category (pooled across items)
    all_ratings = ratings.flatten()
    category_props = np.array([np.mean(all_ratings == k) 
                               for k in range(n_categories)])
    
    # P_e (expected agreement by chance)
    P_e = np.sum(category_props ** 2)
    
    # Calculate P (observed agreement) for each item
    for i in range(n_items):
        item_ratings = ratings[i, :]
        
        # Count in each category
        counts = np.array([np.sum(item_ratings == k) 
                          for k in range(n_categories)])
        
        # P_i for this item
        sum_squares = np.sum(counts ** 2)
        P_i = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
        P_values.append(P_i)
    
    # Average P across items
    P_bar = np.mean(P_values)
    
    # Calculate kappa
    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa

def calculate_item_kappas(ratings, n_categories):
    """Calculate Fleiss' kappa for each item separately."""
    n_items, n_raters = ratings.shape
    results = []
    
    # Global category proportions for P_e
    all_ratings = ratings.flatten()
    category_props = np.array([np.mean(all_ratings == k) 
                               for k in range(n_categories)])
    P_e = np.sum(category_props ** 2)
    
    for i in range(n_items):
        item_ratings = ratings[i, :]
        
        # Counts per category
        counts = np.array([np.sum(item_ratings == k) 
                          for k in range(n_categories)])
        
        # P_i (observed agreement)
        sum_squares = np.sum(counts ** 2)
        P_i = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
        
        # Item kappa
        if P_e < 1.0:
            kappa = (P_i - P_e) / (1 - P_e)
        else:
            kappa = 1.0 if P_i >= 1.0 else 0.0
        
        # Plurality category
        plurality_cat = np.argmax(counts)
        n_plurality = counts[plurality_cat]
        prop_agree = n_plurality / n_raters
        
        results.append({
            'item_id': i,
            'fleiss_kappa': kappa,
            'plurality_cat': int(plurality_cat),
            'n_plurality': int(n_plurality),
            'prop_agree': prop_agree
        })
    
    return pd.DataFrame(results)

# ============================================================================
# RATING SIMULATION 
# ============================================================================

def simulate_expert_ratings_direct(n_raters, n_items, n_categories, 
                                   target_kappa=0.50):
    """Simulate expert ratings using direct agreement probability method."""
    
    P_e = 1.0 / n_categories
    required_P = target_kappa * (1 - P_e) + P_e
    
    # Agreement probability
    agree_prob = np.sqrt(required_P) if required_P >= 0 else 0.0
    agree_prob = np.clip(agree_prob, 1.0/n_categories, 1.0)
    
    # Generate reference categories
    reference_categories = np.random.randint(0, n_categories, size=n_items)
    
    # Generate ratings
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

# ============================================================================
# POWER ANALYSIS (ORIGINAL)
# ============================================================================

def run_single_simulation(n_raters, construct_info, true_kappa=0.50,
                         kappa_threshold=0.40):
    """Run single simulation iteration for power analysis."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    
    # Simulate ratings
    ratings = simulate_expert_ratings_direct(
        n_raters=n_raters,
        n_items=n_items,
        n_categories=n_categories,
        target_kappa=true_kappa
    )
    
    # Calculate overall kappa
    overall_kappa = calculate_fleiss_kappa_from_ratings(ratings, n_categories)
    
    # Calculate item-level kappas
    item_details = calculate_item_kappas(ratings, n_categories)
    
    # Count adequate items
    n_adequate = (item_details['fleiss_kappa'] >= kappa_threshold).sum()
    prop_adequate = n_adequate / n_items
    
    # Statistical test
    se_kappa = np.sqrt(2 / (n_items * n_raters * (n_raters - 1)))
    
    # 95% CI
    ci_lower = overall_kappa - 1.96 * se_kappa
    ci_upper = overall_kappa + 1.96 * se_kappa
    ci_width = ci_upper - ci_lower
    
    # Test H0: kappa = 0
    z_stat = overall_kappa / se_kappa
    p_value = 1 - stats.norm.cdf(z_stat)
    
    # Power
    significant = p_value < 0.05
    meets_threshold = overall_kappa >= kappa_threshold
    achieved_power = 1 if (significant and meets_threshold) else 0
    
    return {
        'n_raters': n_raters,
        'true_kappa': true_kappa,
        'observed_kappa': overall_kappa,
        'kappa_se': se_kappa,
        'kappa_ci_lower': ci_lower,
        'kappa_ci_upper': ci_upper,
        'kappa_ci_width': ci_width,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': significant,
        'meets_threshold': meets_threshold,
        'achieved_power': achieved_power,
        'n_adequate_items': n_adequate,
        'prop_adequate_items': prop_adequate,
        'mean_item_kappa': item_details['fleiss_kappa'].mean(),
        'sd_item_kappa': item_details['fleiss_kappa'].std(),
        'min_item_kappa': item_details['fleiss_kappa'].min(),
        'max_item_kappa': item_details['fleiss_kappa'].max()
    }

def run_simulation_study(constructs=None,
                        n_raters_range=None,
                        true_kappa_range=None,
                        n_iterations=500,
                        kappa_threshold=0.40,
                        n_jobs=-1):
    """Run complete power analysis simulation study."""
    
    if constructs is None:
        constructs = survey_structure
    
    if n_raters_range is None:
        n_raters_range = [8, 10, 12, 15, 18, 20, 25]
    
    if true_kappa_range is None:
        true_kappa_range = [0.40, 0.50, 0.60, 0.70]
    
    # Parameter combinations
    param_combinations = []
    for construct_name in constructs.keys():
        for n_raters in n_raters_range:
            for true_kappa in true_kappa_range:
                for iteration in range(n_iterations):
                    param_combinations.append({
                        'construct': construct_name,
                        'n_raters': n_raters,
                        'true_kappa': true_kappa,
                        'iteration': iteration
                    })
    
    print(f"Running {len(param_combinations)} simulations...")
    print(f"Constructs: {list(constructs.keys())}")
    print(f"Rater sample sizes: {n_raters_range}")
    print(f"True kappa values: {true_kappa_range}")
    print(f"Iterations per condition: {n_iterations}\n")
    
    def run_one_simulation(params):
        construct_info = constructs[params['construct']]
        result = run_single_simulation(
            n_raters=params['n_raters'],
            construct_info=construct_info,
            true_kappa=params['true_kappa'],
            kappa_threshold=kappa_threshold
        )
        result['construct'] = params['construct']
        result['iteration'] = params['iteration']
        return result
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_simulation)(params) 
        for params in tqdm(param_combinations, desc="Power analysis")
    )
    
    return pd.DataFrame(results)

def summarize_simulation_results(sim_results):
    """Summarize power analysis results."""
    
    summary = sim_results.groupby(['construct', 'n_raters', 'true_kappa']).agg({
        'achieved_power': ['mean', 'std'],
        'observed_kappa': ['mean', 'std', 
                          lambda x: x.quantile(0.025), 
                          lambda x: x.quantile(0.975)],
        'kappa_ci_width': ['mean', 'std'],
        'prop_adequate_items': ['mean', 'std'],
        'iteration': 'count'
    }).reset_index()
    
    summary.columns = [
        'construct', 'n_raters', 'true_kappa',
        'empirical_power', 'power_sd',
        'mean_observed_kappa', 'sd_observed_kappa',
        'kappa_2.5_pct', 'kappa_97.5_pct',
        'mean_ci_width', 'sd_ci_width',
        'mean_prop_adequate', 'sd_prop_adequate',
        'n_simulations'
    ]
    
    # Bias and RMSE
    summary['bias'] = summary['mean_observed_kappa'] - summary['true_kappa']
    summary['rmse'] = np.sqrt(summary['bias']**2 + summary['sd_observed_kappa']**2)
    
    # Power CIs
    summary['power_se'] = summary['power_sd'] / np.sqrt(summary['n_simulations'])
    summary['power_ci_lower'] = (summary['empirical_power'] - 
                                 1.96 * summary['power_se']).clip(0, 1)
    summary['power_ci_upper'] = (summary['empirical_power'] + 
                                 1.96 * summary['power_se']).clip(0, 1)
    
    return summary

# ============================================================================
# ENHANCED ANALYSIS 1: STABILITY
# ============================================================================

def analyze_stability(construct_info, true_kappa=0.50, max_raters=30, 
                     n_iterations=100):
    """
    Analyze how kappa estimates stabilize as raters are added incrementally.
    
    Answers: "When do estimates stop changing?"
    """
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    
    results = []
    
    for iteration in tqdm(range(n_iterations), desc="  Stability iterations"):
        # Generate ratings for maximum number of raters
        ratings = simulate_expert_ratings_direct(
            n_raters=max_raters,
            n_items=n_items,
            n_categories=n_categories,
            target_kappa=true_kappa
        )
        
        # Calculate kappa for incrementally larger samples
        kappas_cumulative = []
        
        for n in range(3, max_raters + 1):
            ratings_subset = ratings[:, :n]
            kappa = calculate_fleiss_kappa_from_ratings(ratings_subset, n_categories)
            kappas_cumulative.append(kappa)
        
        # Calculate metrics
        for idx, n in enumerate(range(3, max_raters + 1)):
            kappa = kappas_cumulative[idx]
            
            # Change from previous
            if idx > 0:
                delta_kappa = abs(kappa - kappas_cumulative[idx - 1])
            else:
                delta_kappa = np.nan
            
            # Change from final
            delta_from_final = abs(kappa - kappas_cumulative[-1])
            
            results.append({
                'iteration': iteration,
                'n_raters': n,
                'kappa': kappa,
                'delta_kappa': delta_kappa,
                'delta_from_final': delta_from_final
            })
    
    results_df = pd.DataFrame(results)
    
    # Summarize
    stability_summary = results_df.groupby('n_raters').agg({
        'kappa': ['mean', 'std'],
        'delta_kappa': ['mean', 'std'],
        'delta_from_final': ['mean', 'std']
    }).reset_index()
    
    stability_summary.columns = [
        'n_raters', 
        'mean_kappa', 'sd_kappa',
        'mean_delta', 'sd_delta',
        'mean_delta_from_final', 'sd_delta_from_final'
    ]
    
    return stability_summary

# ============================================================================
# ENHANCED ANALYSIS 2: PRECISION (BOOTSTRAP)
# ============================================================================

def analyze_precision_bootstrap(construct_info, n_raters_range, 
                                true_kappa=0.50, n_iterations=100,
                                n_bootstrap=500):
    """
    Analyze precision using bootstrap confidence intervals.
    
    Answers: "How precise are estimates?"
    """
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    
    results = []
    
    for n_raters in tqdm(n_raters_range, desc="  Bootstrap precision"):
        
        ci_widths = []
        
        for iteration in range(n_iterations):
            # Generate ratings
            ratings = simulate_expert_ratings_direct(
                n_raters=n_raters,
                n_items=n_items,
                n_categories=n_categories,
                target_kappa=true_kappa
            )
            
            # Bootstrap CI
            kappas_boot = []
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n_items, size=n_items, 
                                               replace=True)
                ratings_boot = ratings[boot_indices, :]
                kappa_boot = calculate_fleiss_kappa_from_ratings(
                    ratings_boot, n_categories
                )
                kappas_boot.append(kappa_boot)
            
            # 95% percentile CI
            ci_lower = np.percentile(kappas_boot, 2.5)
            ci_upper = np.percentile(kappas_boot, 97.5)
            ci_width = ci_upper - ci_lower
            
            ci_widths.append(ci_width)
        
        # Summarize
        results.append({
            'n_raters': n_raters,
            'mean_ci_width': np.mean(ci_widths),
            'sd_ci_width': np.std(ci_widths),
            'median_ci_width': np.median(ci_widths)
        })
    
    return pd.DataFrame(results)

# ============================================================================
# ENHANCED ANALYSIS 3: REPLICATION VARIABILITY
# ============================================================================

def analyze_replication_variability(construct_info, n_raters_range, 
                                   true_kappa=0.50, n_iterations=200):
    """
    Analyze variability across independent samples.
    
    Answers: "How much do results vary across replications?"
    
    This is the CORRECTED reliability analysis - focuses on SD and range,
    not correlation/ICC which don't apply to this design.
    """
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    
    results = []
    
    for n_raters in tqdm(n_raters_range, desc="  Replication variability"):
        
        # Generate multiple independent samples
        kappas_all = []
        
        for iteration in range(n_iterations):
            ratings = simulate_expert_ratings_direct(
                n_raters=n_raters,
                n_items=n_items,
                n_categories=n_categories,
                target_kappa=true_kappa
            )
            kappa = calculate_fleiss_kappa_from_ratings(ratings, n_categories)
            kappas_all.append(kappa)
        
        # Calculate variability metrics
        mean_kappa = np.mean(kappas_all)
        sd_across_samples = np.std(kappas_all)
        
        # Coefficient of variation
        cv = sd_across_samples / mean_kappa if mean_kappa > 0 else np.nan
        
        # 95% range of estimates
        percentile_2_5 = np.percentile(kappas_all, 2.5)
        percentile_97_5 = np.percentile(kappas_all, 97.5)
        range_95 = percentile_97_5 - percentile_2_5
        
        # 90% range (less stringent)
        percentile_5 = np.percentile(kappas_all, 5)
        percentile_95 = np.percentile(kappas_all, 95)
        range_90 = percentile_95 - percentile_5
        
        results.append({
            'n_raters': n_raters,
            'mean_kappa': mean_kappa,
            'sd_across_samples': sd_across_samples,
            'cv': cv,
            'range_95': range_95,
            'range_90': range_90,
            'percentile_2_5': percentile_2_5,
            'percentile_97_5': percentile_97_5,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95
        })
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_power_curves(summary_stats, save_path='power_curves.png'):
    """Plot power curves for all constructs."""
    constructs = summary_stats['construct'].unique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, construct in enumerate(constructs):
        ax = axes[idx]
        data = summary_stats[summary_stats['construct'] == construct]
        
        for kappa_val in sorted(data['true_kappa'].unique()):
            kappa_data = data[data['true_kappa'] == kappa_val]
            ax.plot(kappa_data['n_raters'], kappa_data['empirical_power'],
                   marker='o', linewidth=2, markersize=8,
                   label=f'κ = {kappa_val:.2f}')
            ax.fill_between(kappa_data['n_raters'],
                          kappa_data['power_ci_lower'],
                          kappa_data['power_ci_upper'], alpha=0.2)
        
        ax.axhline(y=0.80, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Raters')
        ax.set_ylabel('Statistical Power')
        ax.set_title(construct, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    for idx in range(len(constructs), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Statistical Power to Detect κ ≥ 0.40', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_stability_analysis(stability_summary, construct_name, 
                           save_path='stability_analysis.png'):
    """Plot stability metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Kappa estimate
    ax = axes[0, 0]
    ax.plot(stability_summary['n_raters'], stability_summary['mean_kappa'],
           marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.axhline(y=stability_summary['mean_kappa'].iloc[-1], 
              linestyle='--', color='red', alpha=0.5,
              label=f'Final (n={stability_summary["n_raters"].max()})')
    ax.fill_between(stability_summary['n_raters'],
                    stability_summary['mean_kappa'] - stability_summary['sd_kappa'],
                    stability_summary['mean_kappa'] + stability_summary['sd_kappa'],
                    alpha=0.2, color='steelblue')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean Fleiss\' Kappa')
    ax.set_title('Kappa Estimate by Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Change in estimate
    ax = axes[0, 1]
    ax.plot(stability_summary['n_raters'], stability_summary['mean_delta'],
           marker='o', linewidth=2, markersize=6, color='darkgreen')
    ax.axhline(y=0.01, linestyle='--', color='red', alpha=0.5,
              label='Δκ = 0.01 (stable)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean Change in κ (vs. n-1)')
    ax.set_title('Estimate Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Difference from final
    ax = axes[1, 0]
    ax.plot(stability_summary['n_raters'], 
           stability_summary['mean_delta_from_final'],
           marker='o', linewidth=2, markersize=6, color='purple')
    ax.axhline(y=0.02, linestyle='--', color='red', alpha=0.5,
              label='Within 0.02 of final')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean |Difference from Final|')
    ax.set_title('Convergence to Final Estimate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # SD (precision)
    ax = axes[1, 1]
    ax.plot(stability_summary['n_raters'], stability_summary['sd_kappa'],
           marker='o', linewidth=2, markersize=6, color='coral')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('SD of Kappa Estimates')
    ax.set_title('Precision: Variability Across Iterations')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.suptitle(f'Stability Analysis: {construct_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_precision_bootstrap(precision_df, construct_name,
                             save_path='precision_bootstrap.png'):
    """Plot bootstrap precision analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(precision_df['n_raters'], 
           precision_df['mean_ci_width'],
           marker='o', linewidth=2, markersize=8,
           label='Bootstrap 95% CI width', color='steelblue')
    
    ax.fill_between(precision_df['n_raters'],
                    precision_df['mean_ci_width'] - precision_df['sd_ci_width'],
                    precision_df['mean_ci_width'] + precision_df['sd_ci_width'],
                    alpha=0.2, color='steelblue')
    
    ax.axhline(y=0.10, linestyle='--', color='red', alpha=0.5,
              label='Target (±0.05)')
    ax.axhline(y=0.15, linestyle='--', color='orange', alpha=0.5,
              label='Acceptable (±0.075)')
    
    ax.set_xlabel('Number of Raters', fontsize=12)
    ax.set_ylabel('95% CI Width for Fleiss\' Kappa', fontsize=12)
    ax.set_title(f'Precision Analysis: {construct_name}\nBootstrap CIs',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_replication_variability(replication_df, construct_name,
                                 save_path='replication_variability.png'):
    """Plot replication variability analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SD across samples
    ax = axes[0, 0]
    ax.plot(replication_df['n_raters'], replication_df['sd_across_samples'],
           marker='o', linewidth=2, markersize=6, color='purple')
    ax.axhline(y=0.05, linestyle='--', color='red', alpha=0.5,
              label='Target SD ≤ 0.05')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('SD Across Independent Samples')
    ax.set_title('Between-Sample Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 95% range
    ax = axes[0, 1]
    ax.plot(replication_df['n_raters'], replication_df['range_95'],
           marker='o', linewidth=2, markersize=6, color='coral')
    ax.axhline(y=0.10, linestyle='--', color='red', alpha=0.5,
              label='Target range ≤ 0.10')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('95% Range of Estimates')
    ax.set_title('Expected Variability in Replication (95%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 90% range
    ax = axes[1, 0]
    ax.plot(replication_df['n_raters'], replication_df['range_90'],
           marker='o', linewidth=2, markersize=6, color='teal')
    ax.axhline(y=0.08, linestyle='--', color='red', alpha=0.5,
              label='Target range ≤ 0.08')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('90% Range of Estimates')
    ax.set_title('Expected Variability in Replication (90%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Coefficient of variation
    ax = axes[1, 1]
    ax.plot(replication_df['n_raters'], replication_df['cv'],
           marker='o', linewidth=2, markersize=6, color='darkgreen')
    ax.axhline(y=0.10, linestyle='--', color='red', alpha=0.5,
              label='Target CV ≤ 0.10')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Relative Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.suptitle(f'Replication Variability Analysis: {construct_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================================
# REPORTING
# ============================================================================

def create_power_report(summary_stats):
    """Generate power analysis text report."""
    print("\n" + "="*70)
    print("POWER ANALYSIS RESULTS")
    print("="*70 + "\n")
    
    for construct in summary_stats['construct'].unique():
        print(f"\n{construct}:")
        print("-" * (len(construct) + 1))
        
        construct_data = summary_stats[summary_stats['construct'] == construct]
        
        for kappa_val in sorted(summary_stats['true_kappa'].unique()):
            kappa_data = construct_data[construct_data['true_kappa'] == kappa_val]
            
            print(f"\n  True κ = {kappa_val:.2f}:")
            for _, row in kappa_data.iterrows():
                print(f"    n={row['n_raters']:2d}: "
                      f"Power={row['empirical_power']:.2%}, "
                      f"Observed κ={row['mean_observed_kappa']:.3f} "
                      f"(±{row['sd_observed_kappa']:.3f})")
    
    print("\n" + "="*70)
    print("POWER RECOMMENDATIONS")
    print("="*70 + "\n")
    
    # Find minimum n for 80% power at κ=0.50
    rec_data = summary_stats[summary_stats['true_kappa'] == 0.50]
    rec_by_n = rec_data.groupby('n_raters')['empirical_power'].min().reset_index()
    adequate = rec_by_n[rec_by_n['empirical_power'] >= 0.80]
    
    if len(adequate) > 0:
        min_n = adequate['n_raters'].min()
        print(f"Based on power analysis (κ=0.50):")
        print(f"  Minimum: {min_n} expert raters")
        print(f"  Conservative: {min_n + 3} raters (with buffer)")
    
    print("\n")

# ============================================================================
# RUN ENHANCED ANALYSES FOR SINGLE CONSTRUCT
# ============================================================================

def run_enhanced_analyses(construct_name, true_kappa=0.50):
    """
    Run all enhanced analyses for a single construct.
    
    Returns comprehensive assessment of stability, precision, and replication.
    """
    print("\n" + "="*70)
    print(f"ENHANCED ANALYSES: {construct_name}")
    print("="*70 + "\n")
    
    construct_info = survey_structure[construct_name]
    output_dir = Path(f'analysis_{construct_name.lower()}')
    output_dir.mkdir(exist_ok=True)
    
    # Analysis 1: Stability
    print("1. Stability Analysis")
    stability_summary = analyze_stability(
        construct_info=construct_info,
        true_kappa=true_kappa,
        max_raters=30,
        n_iterations=100
    )
    
    stability_summary.to_csv(output_dir / 'stability_summary.csv', index=False)
    plot_stability_analysis(stability_summary, construct_name,
                           save_path=output_dir / 'stability_analysis.png')
    
    # Find stabilization point
    stable_threshold = 0.01
    stable_rows = stability_summary[stability_summary['mean_delta'] < stable_threshold]
    if len(stable_rows) > 0:
        stable_point = stable_rows['n_raters'].min()
        print(f"  ✓ Estimates stabilize at n = {stable_point} raters")
    else:
        stable_point = 30
        print(f"  ⚠ Estimates did not fully stabilize (need >{stable_point} raters)")
    
    # Analysis 2: Bootstrap Precision
    print("\n2. Bootstrap Precision Analysis")
    precision_df = analyze_precision_bootstrap(
        construct_info=construct_info,
        n_raters_range=[5, 8, 10, 12, 15, 18, 20, 25, 30],
        true_kappa=true_kappa,
        n_iterations=100,
        n_bootstrap=500
    )
    
    precision_df.to_csv(output_dir / 'precision_bootstrap.csv', index=False)
    plot_precision_bootstrap(precision_df, construct_name,
                            save_path=output_dir / 'precision_bootstrap.png')
    
    # Find precision thresholds
    precise_010_rows = precision_df[precision_df['mean_ci_width'] <= 0.10]
    precise_015_rows = precision_df[precision_df['mean_ci_width'] <= 0.15]
    
    if len(precise_010_rows) > 0:
        precise_010 = precise_010_rows['n_raters'].min()
        print(f"  ✓ Precision ±0.05 achieved at n = {precise_010} raters")
    else:
        precise_010 = 30
        print(f"  ⚠ Precision ±0.05 not achieved (need >{precise_010} raters)")
        
    if len(precise_015_rows) > 0:
        precise_015 = precise_015_rows['n_raters'].min()
        print(f"  ✓ Precision ±0.075 achieved at n = {precise_015} raters")
    else:
        precise_015 = 30
        print(f"  ⚠ Precision ±0.075 not achieved")
    
    # Analysis 3: Replication Variability
    print("\n3. Replication Variability Analysis")
    replication_df = analyze_replication_variability(
        construct_info=construct_info,
        n_raters_range=[5, 8, 10, 12, 15, 18, 20, 25, 30],
        true_kappa=true_kappa,
        n_iterations=200
    )
    
    replication_df.to_csv(output_dir / 'replication_variability.csv', index=False)
    plot_replication_variability(replication_df, construct_name,
                                 save_path=output_dir / 'replication_variability.png')
    
    # Find replication thresholds
    sd_rows = replication_df[replication_df['sd_across_samples'] <= 0.05]
    range_rows = replication_df[replication_df['range_95'] <= 0.10]
    
    if len(sd_rows) > 0:
        sd_threshold = sd_rows['n_raters'].min()
        print(f"  ✓ SD ≤ 0.05 achieved at n = {sd_threshold} raters")
    else:
        sd_threshold = 30
        print(f"  ⚠ SD ≤ 0.05 not achieved")
    
    if len(range_rows) > 0:
        range_threshold = range_rows['n_raters'].min()
        print(f"  ✓ 95% range ≤ 0.10 achieved at n = {range_threshold} raters")
    else:
        range_threshold = 30
        print(f"  ⚠ 95% range ≤ 0.10 not achieved")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{construct_name} (κ = {true_kappa}):")
    print(f"  • Stability: {stable_point}+ raters")
    print(f"  • Precision (±0.05): {precise_010}+ raters")
    print(f"  • Replication SD: {sd_threshold}+ raters")
    print(f"  • Replication range: {range_threshold}+ raters")
    
    overall_rec = max(stable_point, precise_010, sd_threshold, range_threshold)
    print(f"\nRecommendation: {overall_rec} raters")
    print(f"Conservative (with buffer): {overall_rec + 3} raters")
    
    print(f"\nResults saved to: {output_dir}/")
    
    return {
        'construct': construct_name,
        'stability': stable_point,
        'precision_tight': precise_010,
        'precision_acceptable': precise_015,
        'sd_threshold': sd_threshold,
        'range_threshold': range_threshold,
        'overall': overall_rec,
        'conservative': overall_rec + 3
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: Power analysis + enhanced analyses for all constructs."""
    print("="*70)
    print("COMPLETE SIMULATION STUDY")
    print("All Constructs - Power, Stability, Precision, Replication")
    print("="*70 + "\n")
    
    output_dir = Path('simulation_results_complete')
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # PART 1: POWER ANALYSIS (ALL CONSTRUCTS)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PART 1: POWER ANALYSIS (ALL CONSTRUCTS)")
    print("="*70 + "\n")
    
    sim_results = run_simulation_study(
        n_raters_range=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        true_kappa_range=[0.40, 0.50, 0.60, 0.70],
        n_iterations=1000,
        n_jobs=-1
    )
    
    sim_results.to_csv(output_dir / 'power_analysis_full.csv', index=False)
    
    summary = summarize_simulation_results(sim_results)
    summary.to_csv(output_dir / 'power_analysis_summary.csv', index=False)
    
    plot_power_curves(summary, save_path=output_dir / 'power_curves_all.png')
    
    create_power_report(summary)
    
    # ========================================================================
    # PART 2: ENHANCED ANALYSES (ALL CONSTRUCTS)
    # ========================================================================
    
    print("\n" + "="*70)
    print("PART 2: ENHANCED ANALYSES (ALL CONSTRUCTS)")
    print("="*70)
    
    all_recommendations = []
    
    for construct_name in survey_structure.keys():
        results = run_enhanced_analyses(
            construct_name=construct_name,
            true_kappa=0.50
        )
        all_recommendations.append(results)
    
    # Consolidated recommendations
    rec_df = pd.DataFrame(all_recommendations)
    rec_df.to_csv(output_dir / 'recommendations_all_constructs.csv', index=False)
    
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS (ALL CONSTRUCTS)")
    print("="*70 + "\n")
    print(rec_df[['construct', 'stability', 'precision_tight', 
                  'range_threshold', 'overall', 'conservative']])
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}/")
    print("\nGenerated:")
    print("  • power_analysis_full.csv")
    print("  • power_analysis_summary.csv")
    print("  • power_curves_all.png")
    print("  • recommendations_all_constructs.csv")
    print("  • analysis_[construct]/ folders for each construct")
    print("\nTotal runtime: ~1-2 hours depending on cores")

# ============================================================================
# QUICK TEST
# ============================================================================

def quick_test():
    """Quick test on Depression only."""
    print("="*70)
    print("QUICK TEST: Depression Only")
    print("="*70 + "\n")
    
    results = run_enhanced_analyses(
        construct_name='Depression',
        true_kappa=0.50
    )
    
    print("\n✓ Test complete! Check 'analysis_depression/' folder.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Run full analysis
        main()
    else:
        # Default: quick test
        print("Running quick test. Use '--full' flag for complete analysis.")
        print("Example: python simulation_expert-mapping_kappa_FINAL.py --full\n")
        quick_test()