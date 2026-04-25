import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde

# Configuration
results_dir = Path("experiment_results_anant")
analysis_base_dir = Path("analysis")
analysis_base_dir.mkdir(exist_ok=True)

def get_result_files(results_dir):
    """Find all individual result JSON files."""
    return sorted(results_dir.glob("results_*.json"))

def extract_task_llm_from_filename(filepath):
    """
    Extract task name and LLM name from filename.
    Format: results_{task}_{llm}_n{N}.json
    """
    stem = filepath.stem  # e.g., "results_hurricane_eal_counties_qwen25_coder_n250"
    
    # Remove "results_" prefix
    parts = stem.replace("results_", "")
    
    # Split by last occurrence of "_n"
    if "_n" in parts:
        name_part, n_part = parts.rsplit("_n", 1)
    else:
        name_part = parts
        n_part = "unknown"
    
    # Now we need to split task from llm
    # Common LLM names: qwen25_coder, gemma4_e4b, etc.
    # Task names: hurricane_eal_counties, inland_flood_eal, wildfire_eal_west
    
    # Strategy: Look for known LLM patterns at the end
    known_llms = ["qwen25_coder", "gemma4_e4b", "qwen25", "gemma4"]
    
    llm_name = None
    task_name = None
    
    for llm in known_llms:
        if name_part.endswith(f"_{llm}"):
            llm_name = llm
            task_name = name_part[:-len(f"_{llm}")]
            break
    
    if llm_name is None:
        # Fallback: assume last two parts are LLM (e.g., "gemma4_e4b")
        parts_split = name_part.split("_")
        if len(parts_split) >= 3:
            llm_name = "_".join(parts_split[-2:])
            task_name = "_".join(parts_split[:-2])
        else:
            llm_name = "unknown"
            task_name = name_part
    
    return task_name, llm_name, n_part


def create_output_dir(task_name, llm_name):
    """Create and return output directory for a specific task/llm combo."""
    output_dir = analysis_base_dir / f"{task_name}_{llm_name}"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def plot_marginal_likelihood_distribution(result, task_name, llm_name, output_dir):
    """
    Plot the distribution of log marginal likelihoods log p(x|m) across models.
    """
    print(f"  Plotting marginal likelihood distribution...")
    
    # Extract log marginal likelihoods
    if "full_result" in result:
        log_marginals = result["full_result"].get("log_marginal_per_model", [])
    else:
        log_marginals = result.get("metrics", {}).get("log_marginal_per_model", [])
    
    # Filter out fallback values
    log_marginals = np.array([lm for lm in log_marginals if lm > -1e10])
    
    if len(log_marginals) == 0:
        print(f"    No valid log marginals found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(
        log_marginals, 
        bins=30, 
        alpha=0.7, 
        color='steelblue',
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add vertical line for mean
    mean_log_marginal = np.mean(log_marginals)
    ax.axvline(mean_log_marginal, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_log_marginal:.2f}')
    
    # Add vertical line for max
    max_log_marginal = np.max(log_marginals)
    ax.axvline(max_log_marginal, color='green', linestyle='--', linewidth=2.5,
               label=f'Max: {max_log_marginal:.2f}')
    
    # Styling
    ax.set_xlabel('log p(x | m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Models', fontsize=12, fontweight='bold')
    ax.set_title(
        f"Marginal Likelihood Distribution\n{task_name} - {llm_name}",
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = (
        f"N models: {len(log_marginals)}\n"
        f"Range: [{np.min(log_marginals):.2f}, {np.max(log_marginals):.2f}]\n"
        f"Std: {np.std(log_marginals):.2f}\n"
        f"Median: {np.median(log_marginals):.2f}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    output_path = output_dir / "marginal_likelihood_distribution.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def plot_bma_vs_loo_weights_scatter(result, task_name, llm_name, output_dir):
    """
    Scatter plot comparing BMA weights vs LOO weights.
    """
    print(f"  Plotting BMA vs LOO weight scatter...")
    
    # Extract weights
    if "full_result" in result:
        weights_bma = np.array(result["full_result"].get("weights_bma", []))
        weights_loo = np.array(result["full_result"].get("weights_loo", []))
    else:
        weights_bma = np.array(result.get("metrics", {}).get("weights_bma", []))
        weights_loo = np.array(result.get("metrics", {}).get("weights_loo", []))
    
    if len(weights_bma) == 0 or len(weights_loo) == 0:
        print(f"    No weights found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot with color gradient
    scatter = ax.scatter(
        weights_bma, weights_loo, 
        s=120, alpha=0.6, 
        c=np.arange(len(weights_bma)), 
        cmap='viridis',
        edgecolors='black', 
        linewidth=1.5
    )
    
    # Add diagonal line (perfect agreement)
    max_weight = max(weights_bma.max(), weights_loo.max())
    ax.plot([0, max_weight], [0, max_weight], 'r--', linewidth=2.5,
           label='Perfect Agreement', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Model Index', fontsize=11)
    
    # Styling
    ax.set_xlabel('BMA Weight', fontsize=12, fontweight='bold')
    ax.set_ylabel('LOO Weight', fontsize=12, fontweight='bold')
    ax.set_title(
        f"BMA vs LOO Weights\n{task_name} - {llm_name}",
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics
    if "metrics" in result:
        l1_distance = result["metrics"].get("l1_distance_loo_bma", 0)
    else:
        l1_distance = np.sum(np.abs(weights_loo - weights_bma))
    
    correlation = np.corrcoef(weights_bma, weights_loo)[0, 1] if len(weights_bma) > 1 else 1.0
    
    stats_text = (
        f"L1 Distance: {l1_distance:.4f}\n"
        f"Correlation: {correlation:.4f}\n"
        f"N models: {len(weights_bma)}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    output_path = output_dir / "bma_vs_loo_weights_scatter.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def plot_weight_distributions(result, task_name, llm_name, output_dir):
    """
    Plot BMA and LOO weight distributions side-by-side.
    """
    print(f"  Plotting weight distributions...")
    
    # Extract weights
    if "full_result" in result:
        weights_bma = np.array(result["full_result"].get("weights_bma", []))
        weights_loo = np.array(result["full_result"].get("weights_loo", []))
    else:
        weights_bma = np.array(result.get("metrics", {}).get("weights_bma", []))
        weights_loo = np.array(result.get("metrics", {}).get("weights_loo", []))
    
    if len(weights_bma) == 0 or len(weights_loo) == 0:
        print(f"    No weights found")
        return
    
    # Sort for better visualization
    weights_bma_sorted = np.sort(weights_bma)[::-1]  # Descending
    weights_loo_sorted = np.sort(weights_loo)[::-1]
    
    model_indices = np.arange(len(weights_bma))
    
    fig, (ax_bma, ax_loo) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BMA plot
    bars_bma = ax_bma.bar(
        model_indices, weights_bma_sorted, 
        color='gold', edgecolor='black', linewidth=1.2,
        alpha=0.8
    )
    
    # Color top 5 bars differently
    for i in range(min(5, len(bars_bma))):
        bars_bma[i].set_color('orange')
    
    ax_bma.set_xlabel('Model Rank', fontsize=12, fontweight='bold')
    ax_bma.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax_bma.set_title(f"BMA Weights", fontsize=13, fontweight='bold')
    ax_bma.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add entropy and ESS
    if "metrics" in result:
        entropy_bma = result["metrics"].get("entropy_bma", 0)
        ess_bma = result["metrics"].get("ess_bma", 0)
    else:
        entropy_bma = -np.sum(weights_bma * np.log(weights_bma + 1e-10))
        ess_bma = 1.0 / np.sum(weights_bma ** 2)
    
    ax_bma.text(
        0.98, 0.98,
        f"Entropy: {entropy_bma:.3f}\nESS: {ess_bma:.2f}",
        transform=ax_bma.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # LOO plot
    bars_loo = ax_loo.bar(
        model_indices, weights_loo_sorted,
        color='lightblue', edgecolor='black', linewidth=1.2,
        alpha=0.8
    )
    
    # Color top 5 bars differently
    for i in range(min(5, len(bars_loo))):
        bars_loo[i].set_color('steelblue')
    
    ax_loo.set_xlabel('Model Rank', fontsize=12, fontweight='bold')
    ax_loo.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax_loo.set_title(f"LOO Weights", fontsize=13, fontweight='bold')
    ax_loo.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add entropy and ESS
    if "metrics" in result:
        entropy_loo = result["metrics"].get("entropy_loo", 0)
        ess_loo = result["metrics"].get("ess_loo", 0)
    else:
        entropy_loo = -np.sum(weights_loo * np.log(weights_loo + 1e-10))
        ess_loo = 1.0 / np.sum(weights_loo ** 2)
    
    ax_loo.text(
        0.98, 0.98,
        f"Entropy: {entropy_loo:.3f}\nESS: {ess_loo:.2f}",
        transform=ax_loo.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    fig.suptitle(f"{task_name} - {llm_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "weight_distributions.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def plot_elbo_convergence(result, task_name, llm_name, output_dir):
    """Plot ELBO convergence for each datapoint."""
    print(f"  Plotting ELBO convergence...")
    
    # Extract LOO diagnostics
    if "full_result" in result:
        loo_diagnostics = result["full_result"].get("loo_diagnostics_per_model", [])
    else:
        loo_diagnostics = result.get("loo_diagnostics_per_model", [])
    
    if not loo_diagnostics:
        print(f"    No LOO diagnostics found")
        return
    
    # Filter for models with true LOO ELBO
    models_with_elbo = []
    for model_idx, diag in enumerate(loo_diagnostics):
        if diag and diag.get('method') == 'true_loo_elbo':
            models_with_elbo.append((model_idx, diag))
    
    if not models_with_elbo:
        print(f"    No models with ELBO histories (using PSIS-LOO)")
        return
    
    print(f"    Found {len(models_with_elbo)} models with ELBO histories")
    
    # Plot for first 3 models
    for model_idx, diag in models_with_elbo[:3]:
        elbo_histories = diag.get('elbo_histories', [])
        n_datapoints = len(elbo_histories)
        
        if n_datapoints == 0:
            continue
        
        # Grid layout
        n_cols = min(4, n_datapoints)
        n_rows = (n_datapoints + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            squeeze=False
        )
        axes = axes.flatten()
        
        for i, elbo_hist in enumerate(elbo_histories):
            if len(elbo_hist) == 0:
                axes[i].text(0.5, 0.5, 'Failed', ha='center', va='center', fontsize=12)
                axes[i].set_title(f'Point {i}', fontweight='bold')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                continue
            
            # Plot raw ELBO samples
            axes[i].plot(elbo_hist, 'o', alpha=0.3, markersize=3, color='steelblue')
            
            # Plot running mean
            running_mean = np.cumsum(elbo_hist) / (np.arange(len(elbo_hist)) + 1)
            axes[i].plot(running_mean, 'r-', linewidth=2, label='Running mean')
            
            # Final estimate line
            final_val = np.mean(elbo_hist)
            axes[i].axhline(final_val, color='green', linestyle='--', linewidth=1.5,
                           label=f'Final: {final_val:.2f}')
            
            # Styling
            axes[i].set_xlabel('MC Iteration', fontsize=9)
            axes[i].set_ylabel('ELBO', fontsize=9)
            axes[i].set_title(f'Datapoint {i}', fontweight='bold', fontsize=10)
            axes[i].legend(fontsize=7, loc='best')
            axes[i].grid(True, alpha=0.2)
            
            # Variance annotation
            std_val = np.std(elbo_hist)
            axes[i].text(0.02, 0.02, f'σ={std_val:.3f}', 
                        transform=axes[i].transAxes,
                        fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
        
        # Hide extra subplots
        for i in range(n_datapoints, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(
            f'ELBO Convergence - Model {model_idx}\n{task_name} - {llm_name}',
            fontsize=12,
            fontweight='bold'
        )
        plt.tight_layout()
        
        output_path = output_dir / f"elbo_convergence_model_{model_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()


def plot_test_elpd_comparison(result, task_name, llm_name, output_dir):
    """Plot test ELPD comparison if available."""
    print(f"  Plotting test ELPD comparison...")
    
    # Extract test ELPD metrics
    if "full_result" in result:
        test_elpd_uniform = result["full_result"].get("test_elpd_uniform")
        test_elpd_bma = result["full_result"].get("test_elpd_bma")
        test_elpd_loo = result["full_result"].get("test_elpd_loo")
    elif "metrics" in result:
        test_elpd_uniform = result["metrics"].get("test_elpd_uniform")
        test_elpd_bma = result["metrics"].get("test_elpd_bma")
        test_elpd_loo = result["metrics"].get("test_elpd_loo")
    else:
        test_elpd_uniform = None
        test_elpd_bma = None
        test_elpd_loo = None
    
    if test_elpd_loo is None:
        print(f"    No test ELPD data found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Uniform', 'BMA', 'LOO']
    values = [test_elpd_uniform, test_elpd_bma, test_elpd_loo]
    colors = ['gray', 'gold', 'steelblue']
    
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Highlight the best
    best_idx = np.argmax(values)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(4)
    
    ax.set_ylabel('Test ELPD', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Test Predictive Performance\n{task_name} - {llm_name}',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{val:.4f}',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=11,
            fontweight='bold'
        )
    
    # Add improvement text
    improvement_vs_bma = test_elpd_loo - test_elpd_bma
    improvement_vs_uniform = test_elpd_loo - test_elpd_uniform
    
    stats_text = (
        f"LOO vs BMA: {improvement_vs_bma:+.4f}\n"
        f"LOO vs Uniform: {improvement_vs_uniform:+.4f}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement_vs_bma > 0 else 'lightcoral', alpha=0.5)
    )
    
    plt.tight_layout()
    output_path = output_dir / "test_elpd_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def process_single_result(result_file):
    """Process a single result file and generate all plots."""
    print(f"\nProcessing: {result_file.name}")
    
    # Extract task and LLM name
    task_name, llm_name, n_models = extract_task_llm_from_filename(result_file)
    print(f"  Task: {task_name}")
    print(f"  LLM: {llm_name}")
    print(f"  N: {n_models}")
    
    # Load result
    with open(result_file) as f:
        result = json.load(f)
    
    if not result.get("success", False):
        print(f"  ❌ Experiment failed, skipping")
        return None
    
    # Create output directory
    output_dir = create_output_dir(task_name, llm_name)
    print(f"  Output dir: {output_dir}")
    
    # Generate all plots
    plot_marginal_likelihood_distribution(result, task_name, llm_name, output_dir)
    plot_bma_vs_loo_weights_scatter(result, task_name, llm_name, output_dir)
    plot_weight_distributions(result, task_name, llm_name, output_dir)
    plot_elbo_convergence(result, task_name, llm_name, output_dir)
    plot_test_elpd_comparison(result, task_name, llm_name, output_dir)
    
    print(f"  ✓ Complete")
    
    return {
        "task": task_name,
        "llm": llm_name,
        "n_models": n_models,
        "result_file": str(result_file),
        "output_dir": str(output_dir)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*80)
    print("LARGE LANGUAGE BAYES - RESULTS ANALYSIS")
    print("="*80)
    
    # Find all result files
    result_files = get_result_files(results_dir)
    print(f"\nFound {len(result_files)} result files")
    
    if len(result_files) == 0:
        print("No result files found!")
        return
    
    # Process each result file
    processed = []
    for result_file in result_files:
        info = process_single_result(result_file)
        if info:
            processed.append(info)
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - Processed {len(processed)} experiments")
    print("="*80)
    
    # Print summary
    print("\nGenerated analysis in:")
    for info in processed:
        print(f"  {info['output_dir']}")
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    for result_file in result_files:
        with open(result_file) as f:
            result = json.load(f)
        
        if not result.get("success"):
            continue
        
        task_name, llm_name, n_models = extract_task_llm_from_filename(result_file)
        metrics = result.get("metrics", {})
        
        row = {
            "Task": task_name,
            "LLM": llm_name,
            "N": n_models,
            "Valid": metrics.get("n_models_valid_final", 0),
            "L1_dist": metrics.get("l1_distance_loo_bma", 0),
            "ESS_BMA": metrics.get("ess_bma", 0),
            "ESS_LOO": metrics.get("ess_loo", 0),
        }
        
        # Add test ELPD if available
        test_elpd_loo = metrics.get("test_elpd_loo")
        test_elpd_bma = metrics.get("test_elpd_bma")
        if test_elpd_loo is not None:
            row["Test_ELPD_LOO"] = test_elpd_loo
            row["Test_ELPD_BMA"] = test_elpd_bma
            row["ELPD_Improvement"] = test_elpd_loo - test_elpd_bma
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save summary
    summary_path = analysis_base_dir / "summary_table.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()