"""
EIML Workshop - Minimal Figure Suite
=====================================

Only 3 things:
1. Predictive performance table (BMA vs LOO)
2. ESS + Entropy comparison 
3. ELBO learning curves (shows it converges)
4. Weight distribution details
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Clean publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 1.2,
})


def load_all_results(results_dir):
    """Load all successful results."""
    results = []
    for filepath in Path(results_dir).glob("results_*.json"):
        with open(filepath) as f:
            result = json.load(f)
        if result.get("success"):
            results.append(result)
    return results


# ============================================================================
# TABLE 1: PREDICTIVE PERFORMANCE
# ============================================================================

def create_predictive_performance_table(results, output_path_csv, output_path_tex):
    """
    Simple table: Dataset | BMA ELPD | LOO ELPD | Improvement
    """
    data = []
    
    for result in results:
        task = result['task']
        llm = result['llm_name']
        metrics = result['metrics']
        
        row = {
            'Dataset': task.replace('_', ' ').title(),
            #'LLM': llm.replace('_', ' ').upper(),
            #'N': metrics['n_models_valid_final'],
        }
        
        # Check if test ELPD available
        if 'test_elpd_loo' in metrics and metrics['test_elpd_loo'] is not None:
            row['BMA ELPD'] = f"{metrics['test_elpd_bma']:.3f}"
            row['LOO ELPD'] = f"{metrics['test_elpd_loo']:.3f}"
            #row['Δ ELPD'] = f"{(metrics['test_elpd_loo'] - metrics['test_elpd_bma']):+.3f}"
        else:
            row['BMA ELPD'] = '—'
            row['LOO ELPD'] = '—'
            #row['Δ ELPD'] = '—'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save CSV
    df.to_csv(output_path_csv, index=False)
    print(f"✓ {output_path_csv.name}")
    
    # Save LaTeX
    latex = df.to_latex(index=False, escape=False, 
                        column_format='llrrrr',
                        caption='Test predictive performance (ELPD). Higher is better.',
                        label='tab:predictive')
    
    with open(output_path_tex, 'w') as f:
        f.write(latex)
    print(f"✓ {output_path_tex.name}")
    
    # Print to console
    print("\nPredictive Performance Table:")
    print(df.to_string(index=False))
    
    return df


# ============================================================================
# FIGURE 1: ESS + ENTROPY COMPARISON
# ============================================================================

def plot_ess_and_entropy(results, output_path):
    """
    Single panel: ESS comparison only.
    Tall and readable.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # Taller figure
    
    # Collect data
    datasets = []
    ess_bma = []
    ess_loo = []
    
    for result in results:
        task = result['task']
        metrics = result['metrics']
        
        # Clean dataset name
        clean_name = task.replace('_', ' ').replace('eal', '').strip().title()
        datasets.append(clean_name)
        ess_bma.append(metrics['ess_bma'])
        ess_loo.append(metrics['ess_loo'])
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # ESS bars
    ax.bar(x - width/2, ess_bma, width, label='BMA', 
           color='#E74C3C', edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, ess_loo, width, label='LOO',
           color='#3498DB', edgecolor='black', linewidth=1.5)
    
    # Styling
    ax.set_ylabel('Effective Sample Size', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=16, rotation=0)  # Increased from 13 to 16
    ax.tick_params(axis='y', labelsize=16)  # Increased from 12 to 16
    ax.tick_params(axis='x', labelsize=16)  # Added explicit x-axis tick size
    ax.legend(frameon=False, loc='upper left', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, max(max(ess_bma), max(ess_loo)) * 1.15])  # Add headroom
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")


# ============================================================================
# FIGURE 2: ELBO LEARNING CURVES
# ============================================================================

def plot_elbo_convergence(result, output_path, show_all=False):
    """
    Show ELBO convergence.
    
    If show_all=False: Show first 6 datapoints (2×3 grid, clean for paper)
    If show_all=True: Show all 20 datapoints (5×4 grid, larger figure for supplementary)
    """
    loo_diagnostics = result['full_result'].get('loo_diagnostics_per_model', [])
    
    if not loo_diagnostics:
        print(f"✗ No LOO diagnostics for ELBO plot")
        return
    
    # Find first model with ELBO histories
    elbo_histories = None
    for diag in loo_diagnostics:
        if diag and diag.get('method') == 'true_loo_elbo':
            elbo_histories = diag.get('elbo_histories', [])
            if elbo_histories:
                break
    
    if not elbo_histories:
        print(f"✗ No ELBO histories found (using PSIS-LOO)")
        return
    
    n_datapoints = len(elbo_histories)
    
    if show_all:
        # Show all 20 in larger format
        n_show = min(20, n_datapoints)
        fig, axes = plt.subplots(5, 4, figsize=(20, 20))  # Much larger!
        axes = axes.flatten()
    else:
        # Show first 6 (clean for paper)
        n_show = min(6, n_datapoints)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
    
    for i in range(max(20 if show_all else 6, n_show)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if i >= n_datapoints:
            ax.axis('off')
            continue
        
        elbo_hist = elbo_histories[i]
        
        if len(elbo_hist) == 0:
            ax.text(0.5, 0.5, 'Failed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='red', fontweight='bold')
            ax.set_title(f'Point {i+1}', fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Plot ELBO samples
        ax.plot(elbo_hist, 'o', alpha=0.25, markersize=4, color='steelblue')
        
        # Running mean
        running_mean = np.cumsum(elbo_hist) / (np.arange(len(elbo_hist)) + 1)
        ax.plot(running_mean, '-', linewidth=2.5, color='red')
        
        # Final value
        final = np.mean(elbo_hist)
        ax.axhline(final, color='green', linestyle='--', linewidth=2)
        
        # Styling
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('ELBO', fontsize=10)
        ax.set_title(f'Point {i+1}', fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add final value as text (cleaner positioning)
        # y_range = ax.get_ylim()
        # ax.text(0.98, 0.98, f'{final:.1f}', 
        #        transform=ax.transAxes, fontsize=10, fontweight='bold',
        #        ha='right', va='top',
        #        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
        #                 edgecolor='darkgreen', linewidth=1.5, alpha=0.8))
    
    # Hide unused subplots
    total_slots = 20 if show_all else 6
    for i in range(n_show, total_slots):
        if i < len(axes):
            axes[i].axis('off')
    
    task_name = result['task'].replace('_', ' ').title()
    #llm_name = result['llm_name'].upper()
    n_shown = min(n_show, 6 if not show_all else 20)
    plt.suptitle(f'ELBO Convergence - {task_name} - {n_shown} Points', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")


# ============================================================================
# TABLE 2: WEIGHT DISTRIBUTION DETAILS
# ============================================================================

def create_weight_distribution_table(result, output_path_csv, output_path_tex):
    """
    Show top-K models and their weights for BMA vs LOO.
    Shows which models each method picks.
    """
    weights_bma = np.array(result['full_result']['weights_bma'])
    weights_loo = np.array(result['full_result']['weights_loo'])
    
    # Get top 3 models by BMA and top 5 by LOO
    top_bma_indices = np.argsort(-weights_bma)[:3] 
    print(top_bma_indices) # Changed: top 3 instead of top 1
    top_loo_indices = np.argsort(-weights_loo)[:5]
    
    # Combine and deduplicate
    all_indices = list(set(list(top_bma_indices) + list(top_loo_indices)))  # Changed: use list
    all_indices.sort(key=lambda i: -max(weights_bma[i], weights_loo[i]))
    
    data = []
    for idx in all_indices[:10]:
        # Determine what this model is
        label = []
        if idx in top_bma_indices:  # Changed: check if in top 3
            label.append('BMA pick')
        if idx in top_loo_indices[:3]:
            label.append('LOO pick')
        label_str = ', '.join(label) if label else ''
        
        data.append({
            'Model': idx,
            'Type': label_str,
            'BMA Weight': f"{weights_bma[idx]:.4f}",
            'LOO Weight': f"{weights_loo[idx]:.4f}",
        })
    
    df = pd.DataFrame(data)
    
    # Save CSV
    df.to_csv(output_path_csv, index=False)
    print(f"✓ {output_path_csv.name}")
    
    # Save LaTeX
    latex = df.to_latex(index=False, escape=False,
                        column_format='llrr',  # Changed: adjusted for 4 columns
                        caption=f'Weight distribution for {result["task"]}. Top 3 BMA models and top 5 LOO models.',
                        label='tab:weights')
    
    with open(output_path_tex, 'w') as f:
        f.write(latex)
    print(f"✓ {output_path_tex.name}")
    
    # Print to console
    print(f"\nWeight Distribution ({result['task']}):")
    print(df.to_string(index=False))
    
    return df


# ============================================================================
# FIGURE 3: WEIGHT DISTRIBUTION PLOT
# ============================================================================

def plot_weight_distribution(result, output_path):
    """
    Simple sorted bar chart showing weight distribution.
    """
    weights_bma = np.array(result['full_result']['weights_bma'])
    weights_loo = np.array(result['full_result']['weights_loo'])
    
    # Sort by LOO weight
    indices = np.argsort(-weights_loo)[:15]  # Top 15
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(indices))
    width = 0.35
    
    ax.bar(x - width/2, weights_bma[indices], width,
          label='BMA', color='#E74C3C', edgecolor='black', linewidth=1)
    ax.bar(x + width/2, weights_loo[indices], width,
          label='LOO', color='#3498DB', edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Weight')
    ax.set_xlabel('Model Rank')
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(indices) + 1))
    ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add ESS annotation
    ess_bma = result['metrics']['ess_bma']
    ess_loo = result['metrics']['ess_loo']
    ax.text(0.98, 0.98, f'ESS: BMA={ess_bma:.2f}, LOO={ess_loo:.2f}',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def generate_workshop_figures(results_dir, output_dir):
    """Generate all workshop figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("EIML WORKSHOP - FIGURE GENERATION")
    print("="*80)
    
    results = load_all_results(results_dir)
    print(f"\nLoaded {len(results)} successful experiments\n")
    
    # Table 1: Predictive Performance
    print("TABLE 1: Predictive Performance")
    create_predictive_performance_table(
        results,
        output_dir / "table_predictive.csv",
        output_dir / "table_predictive.tex"
    )
    
    # Figure 1: ESS + Entropy
    print("\nFIGURE 1: ESS and Entropy Comparison")
    plot_ess_and_entropy(results, output_dir / "fig_ess_entropy.pdf")
    
    # Figure 2: ELBO (use first dataset with ELBO data)
    print("\nFIGURE 2: ELBO Learning Curves (All Datasets)")
    elbo_count = 0
    for result in results:
        task = result['task']
        llm = result['llm_name']
        if result['full_result'].get('loo_diagnostics_per_model'):
            output_file = output_dir / f"fig_elbo_{task}.pdf"
            plot_elbo_convergence(result, output_file)
            elbo_count += 1
    
    # Table 2 + Figure 3: Weight distribution (use first dataset as example)
    # Table 2 + Figure 3: Weight distribution for ALL datasets
    print("\nTABLE 2 & FIGURE 3: Weight Distribution (All Datasets)")
    
    all_weight_data = []
    
    for result in results:
        task = result['task']
        llm = result['llm_name']
        weights_bma = np.array(result['full_result']['weights_bma'])
        weights_loo = np.array(result['full_result']['weights_loo'])
        
        # Get top models
        top_bma_idx = np.argsort(-weights_bma)[:3]
        top_loo_indices = np.argsort(-weights_loo)[:5]
        
        # Add to combined table
        dataset_name = f"{task.replace('_', ' ')}"
        
        # Show top BMA model
        for rank, idx in enumerate(top_bma_idx ,1):
            all_weight_data.append({ 
                'Dataset': dataset_name,
                'Model': idx,
                'Method': f'BMA (rank {rank})',
                'Weight': f"{weights_bma[idx]:.4f}",
                #'ESS': f"{result['metrics']['ess_bma']:.2f}"
            })
        
        # Show top 3 LOO models
        for rank, idx in enumerate(top_loo_indices[:3], 1):
            all_weight_data.append({
                'Dataset': dataset_name,
                'Model': idx,
                'Method': f'LOO (rank {rank})',
                'Weight': f"{weights_loo[idx]:.4f}",
                #'ESS': f"{result['metrics']['ess_loo']:.2f}" if rank == 1 else ''
            })
        
        # Also create individual table for this dataset
        create_weight_distribution_table(
            result,
            output_dir / f"table_weights_{task}_{llm}.csv",
            output_dir / f"table_weights_{task}_{llm}.tex"
        )
        
        # Individual weight plot
        plot_weight_distribution(result, output_dir / f"fig_weights_{task}_{llm}.pdf")
    
    # Save combined table
    df_all_weights = pd.DataFrame(all_weight_data)
    df_all_weights.to_csv(output_dir / "table_weights_all.csv", index=False)
    
    latex_all = df_all_weights.to_latex(
        index=False, escape=False,
        column_format='llllr',
        caption='Weight distribution across all datasets. BMA concentrates on single model (ESS≈1), LOO spreads across multiple models (ESS≈2-3).',
        label='tab:weights_all'
    )
    with open(output_dir / "table_weights_all.tex", 'w') as f:
        f.write(latex_all)
    
    print(f"✓ table_weights_all.csv")
    print(f"✓ table_weights_all.tex")
    print(f"✓ Generated {len(results)} individual weight tables")
    print(f"✓ Generated {len(results)} individual weight plots")
    
    print("\nCombined Weight Table:")
    print(df_all_weights.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ ALL FIGURES GENERATED")
    print(f"Output: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    generate_workshop_figures(
        results_dir="experiment_results_anant",
        output_dir="workshop_final"
    )