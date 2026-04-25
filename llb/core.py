import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import pickle
from pathlib import Path
from scipy.stats import gaussian_kde, norm

from .mcmc_log import estimate_log_marginal_iw, run_inference, estimate_loo_log_likelihoods, _get_num_datapoints
from .llm import LLMClient
from .model_generator import generate_models_with_diagnostics, load_pregenerated_codes

# Initialize rich console
console = Console()


class NoValidModelsError(RuntimeError):
    """Raised when no valid generated models remain for aggregation."""


def _filter_pathological_models(valid_models, final_targets, verbose=False):
    """
    Filter out models with pathological predictions or marginal likelihoods.
    
    Returns:
        filtered_models: List of non-pathological models
        num_dropped: Number of models dropped
    """
    kept = []
    dropped = 0
    
    # Collect all predictions to compute statistics
    all_means = []
    all_log_marginals = []
    
    for model in valid_models:
        target_samples = model["target_samples"]
        log_marg = model.get("log_marginal_bound", -np.inf)
        
        # Compute mean prediction for first target
        if final_targets and final_targets[0] in target_samples:
            samples = np.asarray(target_samples[final_targets[0]], dtype=np.float64)
            if samples.ndim > 0 and samples.shape[0] > 0:
                mean_pred = float(np.mean(samples))
                all_means.append(mean_pred)
                all_log_marginals.append(log_marg)
    
    if not all_means:
        return valid_models, 0
    
    # Compute robust statistics (use median and MAD to handle outliers)
    median_pred = np.median(all_means)
    mad_pred = np.median(np.abs(np.array(all_means) - median_pred))
    
    median_log_marg = np.median([lm for lm in all_log_marginals if np.isfinite(lm)])
    
    # Define thresholds
    # For predictions: keep within median ± 100 * MAD (very generous)
    pred_lower = median_pred - 100 * mad_pred
    pred_upper = median_pred + 100 * mad_pred
    
    # For log marginals: drop if < median - 50 (catastrophically bad)
    log_marg_threshold = median_log_marg - 50
    
    console.print(f"\n[bold cyan]Pathological Model Filtering:[/bold cyan]")
    console.print(f"  Prediction range: [{pred_lower:.2e}, {pred_upper:.2e}]")
    console.print(f"  Log marginal threshold: {log_marg_threshold:.2f}")
    
    for model in valid_models:
        target_samples = model["target_samples"]
        log_marg = model.get("log_marginal_bound", -np.inf)
        
        # Check prediction
        if final_targets and final_targets[0] in target_samples:
            samples = np.asarray(target_samples[final_targets[0]], dtype=np.float64)
            if samples.ndim > 0 and samples.shape[0] > 0:
                mean_pred = float(np.mean(samples))
                
                # Drop if prediction is pathological
                if mean_pred < pred_lower or mean_pred > pred_upper:
                    if verbose:
                        console.print(f"  [yellow]Dropped model: mean={mean_pred:.2e} (pathological prediction)[/yellow]")
                    dropped += 1
                    continue
        
        # Drop if log marginal is catastrophically bad
        if np.isfinite(log_marg) and log_marg < log_marg_threshold:
            if verbose:
                console.print(f"  [yellow]Dropped model: log_marginal={log_marg:.2f} (catastrophically bad)[/yellow]")
            dropped += 1
            continue
        
        kept.append(model)
    
    if dropped > 0:
        console.print(f"  [green]Filtered: kept {len(kept)}/{len(valid_models)} models (dropped {dropped})[/green]")
    
    return kept, dropped


def _solve_stacking_optimization(loo_log_liks_matrix, verbose=False, lambda_reg=0.01, kl_reference='uniform', reference_weights=None, temperature=1.0):
    """
    Solve the stacking optimization problem with log-space parameterization.
    
    Args:
        loo_log_liks_matrix: (n_datapoints, n_models) matrix of LOO log likelihoods
        verbose: Print detailed optimization info
        lambda_reg: Regularization strength
        kl_reference: Which reference to use for KL regularization
                      - 'uniform': KL(uniform || w) - mode-seeking
                      - 'bma': KL(w_bma || w) - mode-seeking (reverse KL)
                      - 'custom': Use provided reference_weights
                      - None or 'none': No regularization (pure stacking)
        reference_weights: Custom reference weights (only used if kl_reference='custom' or 'bma')
        temperature: Temperature for LOO scores (higher = softer, default=1.0)
        
    Note: We use REVERSE KL: KL(ref || w) - mode-seeking behavior
          We optimize in log-space (softmax parameterization) for numerical stability
          We use n_models - 1 free parameters (pin last to 0) to avoid flat direction
    """
    n_datapoints, n_models = loo_log_liks_matrix.shape
    
    if n_models == 1:
        return np.array([1.0])
    
    # TEMPERING: Scale LOO scores to prevent extreme concentration
    # Higher temperature = softer weights
    loo_matrix_tempered = loo_log_liks_matrix / temperature
    
    # Determine reference distribution for regularization
    if kl_reference is None or kl_reference == 'none':
        ref_weights = None
        reg_type = "None"
    elif kl_reference == 'uniform':
        ref_weights = np.ones(n_models) / n_models
        reg_type = "Uniform (reverse KL)"
    elif kl_reference == 'bma':
        if reference_weights is None:
            raise ValueError("reference_weights must be provided when kl_reference='bma'")
        ref_weights = np.asarray(reference_weights, dtype=np.float64)
        ref_weights = ref_weights / np.sum(ref_weights)
        reg_type = "BMA weights (reverse KL)"
    elif kl_reference == 'custom':
        if reference_weights is None:
            raise ValueError("reference_weights must be provided when kl_reference='custom'")
        ref_weights = np.asarray(reference_weights, dtype=np.float64)
        ref_weights = ref_weights / np.sum(ref_weights)
        reg_type = "Custom (reverse KL)"
    else:
        raise ValueError(f"Unknown kl_reference: {kl_reference}. Must be 'uniform', 'bma', 'custom', or None")
    
    console.rule("[bold cyan]Stacking Optimization[/bold cyan]", style="cyan")
    print(f"n_datapoints: {n_datapoints}, n_models: {n_models}")
    print(f"Regularization: λ={lambda_reg}, Reference: {reg_type}")
    print(f"Parameterization: Log-space (softmax), Temperature: {temperature}")
    
    def theta_to_weights(theta_free):
        """Convert n_models-1 free parameters to n_models weights.
        
        Pin last θ to 0 to remove flat direction in softmax.
        This makes optimization more stable.
        """
        theta = np.zeros(n_models, dtype=np.float64)
        theta[:-1] = theta_free  # First n-1 are free
        # theta[-1] = 0.0  (already zero)
        
        # Softmax with numerical stability
        theta_shifted = theta - np.max(theta)
        exp_theta = np.exp(theta_shifted)
        weights = exp_theta / np.sum(exp_theta)
        return weights
    
    def objective(theta_free):
        """Objective in terms of n_models-1 unconstrained parameters."""
        w = theta_to_weights(theta_free)
        
        # Stacking term (using TEMPERED LOO scores)
        log_sum = np.zeros(n_datapoints, dtype=np.float64)
        for i in range(n_datapoints):
            max_val = np.max(loo_matrix_tempered[i, :])
            log_sum[i] = max_val + np.log(
                np.sum(w * np.exp(loo_matrix_tempered[i, :] - max_val))
            )
        
        stacking_term = np.mean(log_sum)
        
        # Reverse KL regularization: KL(ref || w)
        if ref_weights is None:
            reg_term = 0.0
        else:
            kl_div = np.sum(ref_weights * (np.log(ref_weights) - np.log(w)))
            reg_term = lambda_reg * kl_div
        
        return -stacking_term + reg_term
    
    def gradient(theta_free):
        """Gradient w.r.t. n_models-1 free parameters."""
        w = theta_to_weights(theta_free)
        
        # Compute gradient w.r.t. weights first
        grad_w = np.zeros(n_models, dtype=np.float64)
        
        # Stacking gradient (using TEMPERED scores)
        for i in range(n_datapoints):
            max_val = np.max(loo_matrix_tempered[i, :])
            exp_vals = np.exp(loo_matrix_tempered[i, :] - max_val)
            weighted_sum = np.sum(w * exp_vals)
            grad_w += exp_vals / weighted_sum
        
        grad_w = -grad_w / n_datapoints
        
        # Reverse KL gradient: ∂/∂w_k KL(ref || w) = -ref_k / w_k
        if ref_weights is not None:
            kl_grad_w = -ref_weights / w
            grad_w += lambda_reg * kl_grad_w
        
        # Chain rule: softmax Jacobian
        # ∂L/∂θ_k = w_k * (grad_w[k] - <grad_w, w>)
        grad_w_weighted_mean = np.dot(grad_w, w)
        grad_theta_full = w * (grad_w - grad_w_weighted_mean)
        
        # Return gradient only for free parameters (first n-1)
        return grad_theta_full[:-1]
    
    # Initialize at uniform weights: θ_k = 0 → w_k = 1/K
    theta0_free = np.zeros(n_models - 1, dtype=np.float64)
    w0 = theta_to_weights(theta0_free)
    
    console.print(f"Initial weights (uniform): [yellow]{w0[:5]}[/yellow]... (showing first 5)")
    console.print(f"Initial objective value: [cyan]{objective(theta0_free):.10f}[/cyan]")
    initial_grad = gradient(theta0_free)
    console.print(f"Initial gradient norm: [cyan]{np.linalg.norm(initial_grad):.10f}[/cyan]")
    console.print(f"Initial gradient (first 5): [yellow]{initial_grad[:5]}[/yellow]")
    
    if ref_weights is not None:
        initial_kl = np.sum(ref_weights * (np.log(ref_weights) - np.log(w0)))
        console.print(f"Initial KL(ref||w): [cyan]{initial_kl:.6f}[/cyan]")
    
    if np.allclose(initial_grad, 0, atol=1e-8):
        console.print("\n[yellow]⚠️  GRADIENT IS ZERO at uniform weights![/yellow]")
        print("This means uniform weights are locally optimal.")
        console.rule(style="cyan")
        return w0
    
    # Optimize (no constraints or bounds needed!)
    result = minimize(
        objective,
        theta0_free,
        method='BFGS',
        jac=gradient,
        options={'gtol': 1e-9, 'maxiter': 1000, 'disp': verbose}
    )
    
    print("\nOptimization completed:")
    console.print(f"  Success: [green]{result.success}[/green]" if result.success else f"  Success: [red]{result.success}[/red]")
    console.print(f"  Message: [dim]{result.message}[/dim]")
    print(f"  Iterations: {result.nit}")
    console.print(f"  Final objective: [cyan]{result.fun:.10f}[/cyan]")
    console.print(f"  Objective improvement: [green]{objective(theta0_free) - result.fun:.10f}[/green]")
    
    # Convert final θ back to weights
    weights = theta_to_weights(result.x)
    
    console.print(f"  Final weights (first 5): [yellow]{weights[:5]}[/yellow]")
    console.print(f"  Weight range: [[cyan]{weights.min():.6f}[/cyan], [cyan]{weights.max():.6f}[/cyan]]")
    console.print(f"  Weight std: [cyan]{np.std(weights):.6f}[/cyan]")
    console.print(f"  Weight sum (should be 1.0): [cyan]{np.sum(weights):.10f}[/cyan]")
    console.print(f"  Effective sample size (ESS): [cyan]{1.0/np.sum(weights**2):.2f}[/cyan]")
    
    if ref_weights is not None:
        final_kl = np.sum(ref_weights * (np.log(ref_weights) - np.log(weights)))
        console.print(f"  Final KL(ref||w): [cyan]{final_kl:.6f}[/cyan]")
        console.print(f"  KL reduction: [green]{initial_kl - final_kl:+.6f}[/green]")
        
        # Forward KL for comparison
        forward_kl = np.sum(weights * (np.log(weights) - np.log(ref_weights)))
        console.print(f"  [dim]For comparison, KL(w||ref): {forward_kl:.6f}[/dim]")
    
    if np.allclose(weights, w0, atol=1e-6):
        console.print("\n[yellow]⚠️  WARNING: Weights did not change from initial uniform![/yellow]")
        print("Likely cause: All models perform identically on LOO.")
    else:
        console.print(f"\n[green]✓ Weights changed from uniform[/green] (max change: {np.max(np.abs(weights - w0)):.6f})")
    
    console.rule(style="cyan")
    
    return weights


def _solve_stacking_optimization_simple(loo_log_liks_matrix):
    """
    Simpler version using just scipy without gradient.
    """
    n_datapoints, n_models = loo_log_liks_matrix.shape
    
    if n_models == 1:
        return np.array([1.0])
    
    def objective(w):
        """Negative of the stacking objective."""
        log_sum = np.zeros(n_datapoints)
        for i in range(n_datapoints):
            max_val = np.max(loo_log_liks_matrix[i, :])
            log_sum[i] = max_val + np.log(
                np.sum(w * np.exp(loo_log_liks_matrix[i, :] - max_val))
            )
        return -np.mean(log_sum)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    w0 = np.ones(n_models) / n_models
    
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9}
    )
    
    weights = np.maximum(result.x, 0.0)
    return weights / np.sum(weights)

def _compute_test_elpd_for_model(model, posterior_samples, data_train, data_test, target_key, verbose=False):
    """
    Compute test ELPD for a single model using posterior samples.
    
    Args:
        model: NumPyro model function
        posterior_samples: Dict of posterior samples {param_name: array}
        data_train: Training data dict
        data_test: Test data dict  
        target_key: Target variable name
        verbose: Print debug info
    
    Returns:
        float: Mean test ELPD across test points (higher is better)
    """
    from numpyro.infer import Predictive
    import jax
    
    # Get test values
    if target_key not in data_test:
        # Try to find the matching key
        for key in data_test.keys():
            if isinstance(data_test[key], (list, np.ndarray)):
                target_key = key
                break
    
    x_test = np.asarray(data_test[target_key], dtype=float)
    n_test = len(x_test)
    
    if n_test == 0:
        return np.nan
    
    try:
        # Generate posterior predictive samples
        predictive = Predictive(model, posterior_samples)
        predictions = predictive(jax.random.PRNGKey(0), data=data_train)
        
        # Select observation key
        obs_keys = [k for k in predictions.keys() if 'obs' in k.lower()]
        if len(obs_keys) == 0:
            obs_keys = list(predictions.keys())
        obs_key = obs_keys[0]
        
        pred_samples = np.asarray(predictions[obs_key]).ravel()
        pred_samples = pred_samples[np.isfinite(pred_samples)]
        
        if pred_samples.size < 2:
            return np.nan
        
        # Compute log predictive density at each test point using KDE
        if np.std(pred_samples) < 1e-12:
            # Use Gaussian fallback for constant predictions
            mu = float(np.mean(pred_samples))
            sigma = 1e-6
            log_densities = [np.log(max(norm.pdf(x_i, loc=mu, scale=sigma), 1e-300)) 
                            for x_i in x_test]
        else:
            kde = gaussian_kde(pred_samples)
            log_densities = []
            for x_i in x_test:
                try:
                    density = max(float(kde.evaluate([x_i])[0]), 1e-300)
                    log_densities.append(np.log(density))
                except:
                    log_densities.append(np.nan)
        
        return float(np.nanmean(log_densities))
        
    except Exception as e:
        if verbose:
            console.print(f"[yellow]Test ELPD computation failed: {e}[/yellow]")
        return np.nan

def infer(
    text,
    data,
    targets=None,
    test_data=None, 
    cache_dir=None, 
    api_url=None,
    api_key=None,
    api_model=None,
    n_models=2,
    mcmc_num_warmup=50,
    mcmc_num_samples=100,
    random_seed=None,
    llm_timeout=None,
    llm_max_retries=2,
    llm_retry_backoff=2.0,
    log_marginal_num_inner=5,
    log_marginal_num_outer=80,
    loo_num_inner=25,
    loo_num_warmup=50,
    loo_num_samples=100,
    use_true_loo=True,
    loo_lambda_reg=0.01,
    loo_kl_reference='uniform',
    loo_temperature=3.0,
    verbose=False,
    auto_print_result=True,
    preloaded_codes_dir=None,
    preloaded_codes_field="canonical_code",
):
    base_seed = int(random_seed) if random_seed is not None else int(np.random.SeedSequence().generate_state(1)[0])

    if preloaded_codes_dir is not None:
        llm = None
        model_codes, gen_diag = load_pregenerated_codes(
            preloaded_codes_dir,
            n_models=n_models,
            field=preloaded_codes_field,
        )
        if verbose:
            console.print(
                f"[green]Loaded {len(model_codes)} pre-generated codes from "
                f"{preloaded_codes_dir}[/green]"
            )
    else:
        llm_kwargs = {
            "api_url": api_url,
            "api_key": api_key,
            "model": api_model,
            "max_retries": llm_max_retries,
            "retry_backoff": llm_retry_backoff,
        }
        if llm_timeout is not None:
            llm_kwargs["timeout"] = llm_timeout
        llm = LLMClient(**llm_kwargs)
        model_codes, gen_diag = generate_models_with_diagnostics(
            llm,
            text=text,
            data=data,
            targets=targets,
            n_models=n_models,
        )
    
    generated_models = len(model_codes)
    all_generated_codes = model_codes.copy()
    model_codes, deduplicated_models = _dedupe_model_codes(model_codes)

    diagnostics = {
        "requested_models": int(n_models),
        "generated_models": int(generated_models),
        "deduplicated_models": int(deduplicated_models),
        "invalid_models_syntax_or_parsing": int(gen_diag.get("invalid_syntax_parsing_count", 0)),
        "generation_request_failures": int(gen_diag.get("generation_request_failures", 0)),
        "first_failure_reason": _first_request_failure_reason(gen_diag),
        "missing_targets_failures": 0,
        "compile_failures": 0,
        "inference_failures": 0,
        "nonfinite_log_bound_drops": 0,
        "shape_mismatch_drops": 0,
        "pathological_drops": 0,
        "loo_failures": 0,
        "valid_models_final": 0,
    }

    def _evaluate_candidates(codes, start_index, cache_dir=None):
        valid_local = []
        failed_local = []
        auto_targets_local = None

        iterator = tqdm(
            enumerate(codes),
            total=len(codes),
            desc="inference (NUTS+IWAE+LOO)",
            unit="model",
            leave=False,
            dynamic_ncols=True,
        )
        for local_idx, code in iterator:
            idx = start_index + local_idx
            try:
                iterator.set_postfix_str(f"idx={idx} phase=NUTS", refresh=True)
                infer_out = run_inference(
                    code=code,
                    data=data,
                    targets=targets,
                    num_warmup=mcmc_num_warmup,
                    num_samples=mcmc_num_samples,
                    rng_seed=base_seed + idx,
                )

                model_info = {
                    "code": code,
                    "target_samples": infer_out["target_samples"],
                    "available_sites": infer_out["available_sites"],
                    "posterior_samples": infer_out["samples"],  
                }

                if cache_dir is not None:
                    cache_file = Path(cache_dir) / f"model_{idx:04d}_posterior.pkl"
                    cache_file.parent.mkdir(exist_ok=True, parents=True)
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump({
                                "model_index": idx,
                                "posterior_samples": {k: v.tolist() for k, v in infer_out["samples"].items()},
                                "code": code,
                            }, f)
                    except Exception as cache_exc:
                        if verbose:
                            console.print(f"[yellow]Failed to cache model {idx}: {cache_exc}[/yellow]")

                # Always compute marginal likelihood
                try:
                    iterator.set_postfix_str(f"idx={idx} phase=IWAE", refresh=True)
                    log_bound = estimate_log_marginal_iw(
                        model=infer_out["model"],
                        data=data,
                        posterior_samples=infer_out["samples"],
                        num_inner=log_marginal_num_inner,
                        num_outer=log_marginal_num_outer,
                        rng_seed=base_seed + 10_000 + idx,
                    )
                    model_info["log_marginal_bound"] = log_bound
                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]Model {idx}: Marginal likelihood failed ({e})[/yellow]")
                    model_info["log_marginal_bound"] = -np.inf

                # Always compute LOO with diagnostics
                try:
                    iterator.set_postfix_str(f"idx={idx} phase=LOO", refresh=True)
                    loo_result = estimate_loo_log_likelihoods(
                        model=infer_out["model"],
                        data=data,
                        posterior_samples=infer_out["samples"],
                        num_inner=loo_num_inner,
                        num_warmup=loo_num_warmup,
                        num_samples=loo_num_samples,
                        rng_seed=base_seed + 20_000 + idx,
                        use_true_loo=use_true_loo,
                        return_diagnostics=True,
                    )
                    
                    if isinstance(loo_result, dict):
                        model_info["loo_log_liks"] = loo_result['loo_log_liks']
                        model_info["loo_diagnostics"] = loo_result['diagnostics']
                    else:
                        model_info["loo_log_liks"] = loo_result
                        model_info["loo_diagnostics"] = None
                        
                except Exception as loo_exc:
                    if verbose:
                        console.print(f"[yellow]Model {idx}: LOO failed ({loo_exc})[/yellow]")
                    n_data = _get_num_datapoints(data)
                    # USE NAN INSTEAD OF -1e12 SENTINEL
                    model_info["loo_log_liks"] = np.full(n_data, np.nan, dtype=np.float64)
                    model_info["loo_diagnostics"] = {'method': 'failed', 'error': str(loo_exc)}

            except Exception as exc:
                msg = str(exc)
                if msg.startswith("compile_error:"):
                    diagnostics["compile_failures"] += 1
                    failed_local.append((idx, msg))
                else:
                    diagnostics["inference_failures"] += 1
                    failed_local.append((idx, msg if msg.startswith("inference_error:") else f"inference_error: {msg}"))
                continue

            if targets is not None and infer_out["missing_targets"]:
                diagnostics["missing_targets_failures"] += 1
                failed_local.append(
                    (idx, f"missing targets: {', '.join(infer_out['missing_targets'])}")
                )
                continue

            valid_local.append(model_info)

            if targets is None:
                site_set = set(infer_out["target_samples"].keys())
                auto_targets_local = site_set if auto_targets_local is None else (auto_targets_local & site_set)

        return valid_local, failed_local, auto_targets_local

    valid, failed_models, auto_targets = _evaluate_candidates(model_codes, start_index=0, cache_dir=cache_dir)

    if not valid and llm is not None:
        extra_goal = int(n_models)
        if extra_goal > 0:
            extra_codes_raw, extra_gen_diag = generate_models_with_diagnostics(
                llm,
                text=text,
                data=data,
                targets=targets,
                n_models=extra_goal,
            )
            extra_generated_models = len(extra_codes_raw)
            extra_codes, extra_deduplicated = _dedupe_model_codes(extra_codes_raw)
            diagnostics["requested_models"] += int(extra_goal)
            diagnostics["generated_models"] += int(extra_generated_models)
            diagnostics["deduplicated_models"] += int(extra_deduplicated)
            diagnostics["invalid_models_syntax_or_parsing"] += int(extra_gen_diag.get("invalid_syntax_parsing_count", 0))
            diagnostics["generation_request_failures"] += int(extra_gen_diag.get("generation_request_failures", 0))
            if diagnostics["first_failure_reason"] is None:
                diagnostics["first_failure_reason"] = _first_request_failure_reason(extra_gen_diag)
            extra_valid, extra_failed, extra_auto_targets = _evaluate_candidates(
                extra_codes,
                start_index=len(model_codes),
                cache_dir=cache_dir
            )
            valid.extend(extra_valid)
            failed_models.extend(extra_failed)
            if targets is None:
                if auto_targets is None:
                    auto_targets = extra_auto_targets
                elif extra_auto_targets is not None:
                    auto_targets = auto_targets & extra_auto_targets

    if not valid:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

    if failed_models and verbose:
        print(f"Skipped {len(failed_models)} invalid model(s) during inference.")

    final_targets = list(targets) if targets is not None else sorted(auto_targets or [])
    if not final_targets:
        raise RuntimeError("No target variables are available in valid inferred models.")

    valid, shape_drops = _filter_models_by_target_shape(valid, final_targets)
    diagnostics["shape_mismatch_drops"] = int(shape_drops)
    if len(valid) == 0:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))
    
    # ========================================
    # FILTER PATHOLOGICAL MODELS
    # ========================================
    valid, pathological_drops = _filter_pathological_models(valid, final_targets, verbose=verbose)
    diagnostics["pathological_drops"] = int(pathological_drops)
    
    if len(valid) == 0:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))
    
    # ========================================
    # METHOD 1: Marginal Likelihood (BMA-style)
    # ========================================
    log_bounds = np.array([v["log_marginal_bound"] for v in valid], dtype=np.float64)
    finite_mask = np.isfinite(log_bounds)
    log_bounds_filtered = log_bounds[finite_mask]
    
    if len(log_bounds_filtered) > 0:
        weights_bma = _softmax_from_logs(log_bounds_filtered)
        weights_bma_full = np.zeros(len(valid))
        weights_bma_full[finite_mask] = weights_bma
        weights_bma_full = weights_bma_full / np.sum(weights_bma_full)
    else:
        console.print("[red]All marginal likelihood bounds are non-finite![/red]")
        weights_bma_full = np.ones(len(valid)) / len(valid)
    
    # ========================================
    # METHOD 2: LOO Stacking
    # ========================================
    loo_matrix_raw = np.column_stack([v["loo_log_liks"] for v in valid])

    loo_matrix_raw[np.abs(loo_matrix_raw) > 1e6] = np.nan
    
    # Drop ANY model with ANY failed LOO entry (NaN)
    valid_models_mask = np.all(np.isfinite(loo_matrix_raw), axis=0)
    loo_matrix = loo_matrix_raw[:, valid_models_mask]
    
    num_dropped_loo = int(np.sum(~valid_models_mask))
    diagnostics["loo_failures"] = num_dropped_loo
    
    if num_dropped_loo > 0:
        console.print(f"[yellow]LOO: Dropped {num_dropped_loo} models with failed LOO computation[/yellow]")
    
    if loo_matrix.shape[1] > 0:
        # Matrix stabilization: row-wise centering
        loo_matrix = loo_matrix - np.max(loo_matrix, axis=1, keepdims=True)
        
        # Diagnostic
        console.rule("[bold magenta]LOO Matrix Diagnostic[/bold magenta]", style="magenta")
        console.print(f"Shape: [cyan]{loo_matrix.shape}[/cyan] (after dropping {num_dropped_loo} models)")
        console.print(f"Temperature: [cyan]{loo_temperature}[/cyan]")
        print(f"\nFirst 3 rows (first 3 datapoints, after centering):")
        print(loo_matrix[:3, :])
        print(f"\nVariance per datapoint (across models):")
        for i in range(min(6, loo_matrix.shape[0])):
            variance = np.var(loo_matrix[i, :])
            range_val = np.max(loo_matrix[i, :]) - np.min(loo_matrix[i, :])
            console.print(f"  Point {i}: var=[yellow]{variance:.8f}[/yellow], range=[green]{range_val:.8f}[/green]")
        console.print(f"\nOverall variance: [yellow]{np.var(loo_matrix):.8f}[/yellow]")
        console.print(f"Overall range: [green]{np.max(loo_matrix) - np.min(loo_matrix):.8f}[/green]")
        
        # Check if all columns are identical
        if loo_matrix.shape[1] > 1:
            first_col = loo_matrix[:, 0]
            all_same = True
            for j in range(1, loo_matrix.shape[1]):
                if not np.allclose(loo_matrix[:, j], first_col, rtol=1e-5):
                    all_same = False
                    break
            
            if all_same:
                console.print("\n[red]⚠️  WARNING: ALL MODELS HAVE IDENTICAL LOO VALUES![/red]")
                print("This means the LOO computation is broken, not the optimization.")
            else:
                console.print(f"\n[green]✓ Models have different LOO values[/green]")
        
        console.rule(style="magenta")
        
        # Optimize
        weights_loo_subset = _solve_stacking_optimization(
            loo_matrix, 
            verbose=verbose, 
            lambda_reg=loo_lambda_reg,
            kl_reference=loo_kl_reference,
            reference_weights=weights_bma_full[valid_models_mask] if loo_kl_reference == 'bma' else None,
            temperature=loo_temperature,
        )
        
        # Map back to full valid set
        weights_loo_full = np.zeros(len(valid))
        weights_loo_full[valid_models_mask] = weights_loo_subset
        weights_loo_full = weights_loo_full / np.sum(weights_loo_full)
    else:
        console.print("[red]All models failed LOO computation![/red]")
        weights_loo_full = np.ones(len(valid)) / len(valid)
    
    # ========================================
    # Compute final LOO objective value
    # ========================================
    final_loo_objective = 0.0
    if loo_matrix.shape[1] > 0:
        for i in range(loo_matrix.shape[0]):
            max_val = np.max(loo_matrix[i, :])
            log_sum_i = max_val + np.log(
                np.sum(weights_loo_full[valid_models_mask] * np.exp(loo_matrix[i, :] - max_val))
            )
            final_loo_objective += log_sum_i
        final_loo_objective /= loo_matrix.shape[0]

    # ========================================
    # COMPUTE TEST ELPD
    # ========================================
    test_elpd_per_model = []
    test_elpd_bma = None
    test_elpd_loo = None
    test_elpd_uniform = None

    if test_data is not None and len(valid) > 0:
        console.rule("[bold cyan]Computing Test ELPD[/bold cyan]", style="cyan")
        
        # Determine target key in test data
        test_target_key = final_targets[0] if final_targets else None
        if test_target_key and test_target_key not in test_data:
            # Try to find matching key
            for key in test_data.keys():
                if isinstance(test_data[key], (list, np.ndarray)):
                    test_target_key = key
                    break
        
        if test_target_key:
            n_test = len(test_data[test_target_key])
            console.print(f"Test data: {n_test} points, Target: {test_target_key}")
            
            # Progress bar for test ELPD
            test_bar = tqdm(
                enumerate(valid),
                total=len(valid),
                desc="Test ELPD",
                unit="model",
                leave=False,
                dynamic_ncols=True,
            )
            
            for i, model_info in test_bar:
                try:
                    # Reconstruct model
                    exec_globals = {}
                    exec(model_info["code"], exec_globals)
                    model_fn = exec_globals.get("model")
                    
                    if model_fn is None:
                        test_elpd_per_model.append(np.nan)
                        continue
                    
                    # Use cached posterior samples (already in memory from inference)
                    posterior = model_info.get("posterior_samples")
                    
                    if posterior is None:
                        test_elpd_per_model.append(np.nan)
                        continue
                    
                    # Compute test ELPD for this model
                    test_elpd_i = _compute_test_elpd_for_model(
                        model_fn, 
                        posterior, 
                        data, 
                        test_data, 
                        test_target_key,
                        verbose=verbose
                    )
                    test_elpd_per_model.append(test_elpd_i)
                    
                    test_bar.set_postfix_str(
                        f"ELPD={test_elpd_i:.3f}" if np.isfinite(test_elpd_i) else "ELPD=NaN"
                    )
                    
                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]Test ELPD failed for model {i}: {e}[/yellow]")
                    test_elpd_per_model.append(np.nan)
            
            test_bar.close()
            
            # Compute weighted test ELPDs
            test_elpd_array = np.array(test_elpd_per_model)
            valid_mask = np.isfinite(test_elpd_array)
            
            if np.sum(valid_mask) > 0:
                # Normalize weights for valid models only
                weights_bma_valid = weights_bma_full[valid_mask]
                weights_bma_valid = weights_bma_valid / np.sum(weights_bma_valid)
                
                weights_loo_valid = weights_loo_full[valid_mask]
                weights_loo_valid = weights_loo_valid / np.sum(weights_loo_valid)
                
                test_elpd_bma = float(np.sum(weights_bma_valid * test_elpd_array[valid_mask]))
                test_elpd_loo = float(np.sum(weights_loo_valid * test_elpd_array[valid_mask]))
                test_elpd_uniform = float(np.mean(test_elpd_array[valid_mask]))
                
                console.print(f"\n[bold]Test ELPD Results:[/bold]")
                console.print(f"  Models with valid test ELPD: [green]{np.sum(valid_mask)}/{len(valid)}[/green]")
                console.print(f"  Uniform:  [dim]{test_elpd_uniform:.4f}[/dim]")
                console.print(f"  BMA:      [yellow]{test_elpd_bma:.4f}[/yellow]")
                console.print(f"  LOO:      [green]{test_elpd_loo:.4f}[/green]")
                
                # Highlight best
                best_elpd = max(test_elpd_uniform, test_elpd_bma, test_elpd_loo)
                if test_elpd_loo == best_elpd:
                    console.print(f"  [bold green]✓ LOO achieves best test ELPD[/bold green]")
                elif test_elpd_bma == best_elpd:
                    console.print(f"  [bold yellow]✓ BMA achieves best test ELPD[/bold yellow]")
                else:
                    console.print(f"  [bold dim]✓ Uniform achieves best test ELPD[/bold dim]")
            else:
                console.print("[red]All test ELPD computations failed[/red]")
        else:
            console.print("[yellow]Could not determine test target key[/yellow]")
        
        console.rule(style="cyan")
    
    # ========================================
    # COMPARISON TABLE
    # ========================================
    console.rule("[bold cyan]Weight Comparison: BMA vs LOO Stacking[/bold cyan]", style="cyan")
    
    comparison_table = Table(
        title="Model Weights Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    comparison_table.add_column("Model", style="dim", justify="right")
    comparison_table.add_column("BMA Weight", style="yellow", justify="right")
    comparison_table.add_column("LOO Weight", style="green", justify="right")
    comparison_table.add_column("Difference", style="magenta", justify="right")
    comparison_table.add_column("Log Marginal", style="cyan", justify="right")
    comparison_table.add_column("LOO Status", style="dim", justify="center")
    
    for i in range(len(valid)):
        w_bma = weights_bma_full[i]
        w_loo = weights_loo_full[i]
        diff = w_loo - w_bma
        log_marg = log_bounds[i]
        loo_ok = "✓" if valid_models_mask[i] else "✗"
        
        # Color code difference
        if abs(diff) > 0.1:
            diff_str = f"[bold magenta]{diff:+.6f}[/bold magenta]"
        elif abs(diff) > 0.01:
            diff_str = f"[magenta]{diff:+.6f}[/magenta]"
        else:
            diff_str = f"[dim]{diff:+.6f}[/dim]"
        
        comparison_table.add_row(
            str(i),
            f"{w_bma:.6f}",
            f"{w_loo:.6f}",
            diff_str,
            f"{log_marg:.2f}" if np.isfinite(log_marg) else "—",
            loo_ok
        )
    
    console.print(comparison_table)
    
    # Summary statistics
    console.print(f"\n[bold]Weight Statistics:[/bold]")
    console.print(f"  Models with valid LOO: [green]{np.sum(valid_models_mask)}/{len(valid)}[/green]")
    console.print(f"  BMA entropy: [yellow]{-np.sum(weights_bma_full * np.log(weights_bma_full + 1e-10)):.4f}[/yellow]")
    console.print(f"  LOO entropy: [green]{-np.sum(weights_loo_full * np.log(weights_loo_full + 1e-10)):.4f}[/green]")
    console.print(f"  BMA max weight: [yellow]{weights_bma_full.max():.6f}[/yellow]")
    console.print(f"  LOO max weight: [green]{weights_loo_full.max():.6f}[/green]")
    console.print(f"  BMA ESS: [yellow]{1.0/np.sum(weights_bma_full**2):.2f}[/yellow]")
    console.print(f"  LOO ESS: [green]{1.0/np.sum(weights_loo_full**2):.2f}[/green]")
    console.print(f"  L1 distance: [magenta]{np.sum(np.abs(weights_loo_full - weights_bma_full)):.6f}[/magenta]")
    console.print(f"  Final LOO objective: [green]{final_loo_objective:.4f}[/green]")
    console.rule(style="cyan")
    
    # ========================================
    # Compute posteriors for BOTH methods
    # ========================================
    diagnostics["valid_models_final"] = int(len(valid))
    
    per_model_target_samples = []
    valid_after_shape = []
    kept_weights_bma = []
    kept_weights_loo = []
    
    for m_idx, model_info in enumerate(valid):
        ok, payload = _normalize_target_sample_map(
            target_samples=model_info["target_samples"],
            targets=final_targets,
        )
        if not ok:
            diagnostics["shape_mismatch_drops"] += 1
            continue
        per_model_target_samples.append(payload)
        valid_after_shape.append(model_info)
        kept_weights_bma.append(float(weights_bma_full[m_idx]))
        kept_weights_loo.append(float(weights_loo_full[m_idx]))

    if len(per_model_target_samples) == 0:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

    valid = valid_after_shape
    weights_bma = np.asarray(kept_weights_bma, dtype=np.float64)
    weights_bma = weights_bma / np.sum(weights_bma)
    weights_loo = np.asarray(kept_weights_loo, dtype=np.float64)
    weights_loo = weights_loo / np.sum(weights_loo)
    flat_weights = np.ones(len(per_model_target_samples)) / len(per_model_target_samples)
    diagnostics["valid_models_final"] = int(len(valid))

    report_targets = _resolve_report_targets(per_model_target_samples, final_targets)

    # Compute epistemic uncertainty for BOTH
    epistemic_uncertainty_bma = {}
    epistemic_uncertainty_loo = {}
    epistemic_uncertainty_uniform = {}

    for target in final_targets:
        mu_per_model = []
        for model_samples in per_model_target_samples:
            arr = np.asarray(model_samples[target], dtype=np.float64)
            mu_per_model.append(_target_mean(arr))

        mu_stack = np.stack(
            [np.asarray(mu, dtype=np.float64) for mu in mu_per_model],
            axis=0,
        )

        # BMA epistemic uncertainty
        mu_bma = np.tensordot(weights_bma, mu_stack, axes=(0, 0))
        S_bma = np.tensordot(weights_bma, (mu_stack - mu_bma) ** 2, axes=(0, 0))
        C_bma = 1.0 - float(np.sum(weights_bma ** 2))
        epistemic_uncertainty_bma[target] = S_bma / C_bma if C_bma > 0 else np.zeros_like(mu_bma)

        # LOO epistemic uncertainty
        mu_loo = np.tensordot(weights_loo, mu_stack, axes=(0, 0))
        S_loo = np.tensordot(weights_loo, (mu_stack - mu_loo) ** 2, axes=(0, 0))
        C_loo = 1.0 - float(np.sum(weights_loo ** 2))
        epistemic_uncertainty_loo[target] = S_loo / C_loo if C_loo > 0 else np.zeros_like(mu_loo)

        mu_uniform = np.tensordot(flat_weights, mu_stack, axes=(0, 0))
        S_uniform = np.tensordot(flat_weights, (mu_stack - mu_uniform) ** 2, axes=(0, 0))
        C_uniform = 1.0 - float(np.sum(flat_weights ** 2))
        epistemic_uncertainty_uniform[target] = S_uniform / C_uniform if C_uniform > 0 else np.zeros_like(mu_uniform)

    # ========================================
    # PRINT DIAGNOSTICS
    # ========================================
    if auto_print_result:
        console.rule("[bold blue]Inference Diagnostics[/bold blue]", style="blue")
        console.print(f"Number of requested models: [cyan]{diagnostics['requested_models']}[/cyan]")
        console.print(f"Number of generated models: [cyan]{diagnostics['generated_models']}[/cyan]")
        console.print(f"Number of deduplicated models: [yellow]{diagnostics['deduplicated_models']}[/yellow]" if diagnostics['deduplicated_models'] > 0 else f"Number of deduplicated models: [dim]{diagnostics['deduplicated_models']}[/dim]")
        console.print(f"Number of invalid models (syntax/parsing): [yellow]{diagnostics['invalid_models_syntax_or_parsing']}[/yellow]" if diagnostics['invalid_models_syntax_or_parsing'] > 0 else f"Number of invalid models (syntax/parsing): [dim]{diagnostics['invalid_models_syntax_or_parsing']}[/dim]")
        console.print(f"Number of generation request failures: [red]{diagnostics['generation_request_failures']}[/red]" if diagnostics['generation_request_failures'] > 0 else f"Number of generation request failures: [dim]{diagnostics['generation_request_failures']}[/dim]")
        console.print(f"Number of models missing required targets: [yellow]{diagnostics['missing_targets_failures']}[/yellow]" if diagnostics['missing_targets_failures'] > 0 else f"Number of models missing required targets: [dim]{diagnostics['missing_targets_failures']}[/dim]")
        console.print(f"Number of models that failed to compile: [red]{diagnostics['compile_failures']}[/red]" if diagnostics['compile_failures'] > 0 else f"Number of models that failed to compile: [dim]{diagnostics['compile_failures']}[/dim]")
        console.print(f"Number of models that failed during inference: [red]{diagnostics['inference_failures']}[/red]" if diagnostics['inference_failures'] > 0 else f"Number of models that failed during inference: [dim]{diagnostics['inference_failures']}[/dim]")
        console.print(f"Number of models dropped due to target shape mismatch: [yellow]{diagnostics['shape_mismatch_drops']}[/yellow]" if diagnostics['shape_mismatch_drops'] > 0 else f"Number of models dropped due to target shape mismatch: [dim]{diagnostics['shape_mismatch_drops']}[/dim]")
        console.print(f"Number of pathological models dropped: [yellow]{diagnostics['pathological_drops']}[/yellow]" if diagnostics['pathological_drops'] > 0 else f"Number of pathological models dropped: [dim]{diagnostics['pathological_drops']}[/dim]")
        console.print(f"Number of models with failed LOO: [yellow]{diagnostics['loo_failures']}[/yellow]" if diagnostics['loo_failures'] > 0 else f"Number of models with failed LOO: [dim]{diagnostics['loo_failures']}[/dim]")
        console.print(f"Number of valid models used in final aggregation: [bold green]{diagnostics['valid_models_final']}[/bold green]")
        
        # Print both weight summaries
        for target in report_targets:
            per_model_samples = [
                np.asarray(model_samples[target], dtype=np.float64)
                for model_samples in per_model_target_samples
            ]
            console.rule(f"[bold green]Model Averaging: {target}[/bold green]", style="green")
            _print_dual_model_averaging_summary(
                samples=per_model_samples,
                weights_bma=weights_bma,
                weights_loo=weights_loo,
                target_name=target,
            )
        
        # Epistemic uncertainty comparison
        console.rule("[bold magenta]Epistemic Uncertainty Comparison[/bold magenta]", style="magenta")
        
        epi_table = Table(
            title="Between-Model Variance (Unbiased)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        epi_table.add_column("Target", style="cyan")
        epi_table.add_column("BMA Epistemic Var", style="yellow", justify="right")
        epi_table.add_column("LOO Epistemic Var", style="green", justify="right")
        epi_table.add_column("Difference", style="magenta", justify="right")
        
        for target in final_targets:
            epi_bma = np.asarray(epistemic_uncertainty_bma[target], dtype=np.float64)
            epi_loo = np.asarray(epistemic_uncertainty_loo[target], dtype=np.float64)
            
            if epi_bma.ndim == 0 and epi_loo.ndim == 0:
                diff = float(epi_loo) - float(epi_bma)
                epi_table.add_row(
                    target,
                    f"{float(epi_bma):.6f}",
                    f"{float(epi_loo):.6f}",
                    f"{diff:+.6f}"
                )
            else:
                epi_table.add_row(
                    target,
                    f"shape: {epi_bma.shape}",
                    f"shape: {epi_loo.shape}",
                    "—"
                )
        
        console.print(epi_table)
        console.print()
    
    # Compute posteriors
    draws_per_model = len(per_model_target_samples[0][final_targets[0]])
    total_draws = draws_per_model * len(per_model_target_samples)
    
    posterior_bma = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=weights_bma,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed),
    )
    
    posterior_loo = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=weights_loo,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed + 1),
    )

    posterior_flat = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=np.ones(len(per_model_target_samples)) / len(per_model_target_samples),
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed + 2),
    )

    if auto_print_result:
        returned_targets = _resolve_report_targets([posterior_loo], report_targets)
        _print_posterior_comparison(posterior_bma, posterior_loo, posterior_flat, returned_targets)
    
    return {
        "posterior_bma": posterior_bma,
        "posterior_loo": posterior_loo,
        "posterior_flat": posterior_flat,
        "epistemic_uncertainty_bma": epistemic_uncertainty_bma,
        "epistemic_uncertainty_loo": epistemic_uncertainty_loo,
        "epistemic_uncertainty_uniform": epistemic_uncertainty_uniform,
        "weights_bma": weights_bma,
        "weights_loo": weights_loo,
        "weights_uniform": flat_weights,
        "diagnostics": diagnostics,
        "final_loo_objective": float(final_loo_objective),
        "log_marginal_per_model": log_bounds.tolist(),
        "loo_diagnostics_per_model": [v.get("loo_diagnostics") for v in valid],
        "test_elpd_per_model": test_elpd_per_model if test_data else None,  
        "test_elpd_bma": test_elpd_bma,  
        "test_elpd_loo": test_elpd_loo,  
        "test_elpd_uniform": test_elpd_uniform,  
        "cache_dir": str(cache_dir) if cache_dir else None,  
        "model_codes": all_generated_codes,
    }

def _print_dual_model_averaging_summary(samples, weights_bma, weights_loo, target_name):
    """Print comparison of BMA vs LOO weights."""
    if len(samples) == 0:
        print("Number of models: 0")
        return

    mu_per_model = [np.asarray(_target_mean(np.asarray(s, dtype=np.float64)), dtype=np.float64) for s in samples]

    # Handle shape mismatches
    shape_groups = {}
    for idx, mu in enumerate(mu_per_model):
        key = tuple(mu.shape)
        shape_groups.setdefault(key, []).append(idx)

    dominant_shape = max(shape_groups, key=lambda k: len(shape_groups[k]))
    keep_idx = shape_groups[dominant_shape]
    dropped_idx = [i for i in range(len(mu_per_model)) if i not in keep_idx]

    mu_stack = np.stack([mu_per_model[i] for i in keep_idx], axis=0)
    kept_weights_bma = np.asarray([weights_bma[i] for i in keep_idx], dtype=np.float64)
    kept_weights_bma = kept_weights_bma / np.sum(kept_weights_bma)
    kept_weights_loo = np.asarray([weights_loo[i] for i in keep_idx], dtype=np.float64)
    kept_weights_loo = kept_weights_loo / np.sum(kept_weights_loo)

    mu_flat = np.mean(mu_stack, axis=0)
    mu_bma = np.tensordot(kept_weights_bma, mu_stack, axes=(0, 0))
    mu_loo = np.tensordot(kept_weights_loo, mu_stack, axes=(0, 0))
    
    console.print(f"Number of models: {len(mu_per_model)}")
    if dropped_idx:
        shape_counts = {str(k): len(v) for k, v in shape_groups.items()}
        console.print(f"[yellow]Shape mismatch detected; using dominant shape {dominant_shape}.[/yellow]")
        print(f"Shape counts: {shape_counts}")
    
    if mu_flat.ndim == 0:
        console.print(f"Flat mean: [dim]{float(mu_flat):.6f}[/dim]")
        console.print(f"BMA mean: [yellow]{float(mu_bma):.6f}[/yellow]")
        console.print(f"LOO mean: [green]{float(mu_loo):.6f}[/green]")
        diff = float(mu_loo) - float(mu_bma)
        console.print(f"Difference (LOO - BMA): [magenta]{diff:+.6f}[/magenta]")
    else:
        console.print(f"Flat mean shape: {mu_flat.shape}")
        console.print(f"BMA mean shape: {mu_bma.shape}")
        console.print(f"LOO mean shape: {mu_loo.shape}")
    
    print()
    
    # Top models table
    console.print("[bold]Top 5 models by LOO weight:[/bold]")
    ranked_loo = np.argsort(-kept_weights_loo)
    for local_idx in ranked_loo[:5]:
        rank_idx = keep_idx[int(local_idx)]
        w_bma = float(weights_bma[rank_idx])
        w_loo = float(weights_loo[rank_idx])
        mu_i = np.asarray(mu_per_model[rank_idx], dtype=np.float64)
        
        if mu_i.ndim == 0:
            console.print(f"  model={rank_idx}, BMA=[yellow]{w_bma:.6f}[/yellow], LOO=[green]{w_loo:.6f}[/green], mu_i={float(mu_i):.6f}")
        else:
            preview = mu_i.reshape(-1)[:3].tolist()
            console.print(f"  model={rank_idx}, BMA=[yellow]{w_bma:.6f}[/yellow], LOO=[green]{w_loo:.6f}[/green], mu_i={preview}...")
    
    print()


def _print_posterior_comparison(posterior_bma, posterior_loo, posterior_flat, targets):
    """Compare posterior means across all three methods."""
    console.rule("[bold blue]Posterior Mean Comparison[/bold blue]", style="blue")
    
    comp_table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    comp_table.add_column("Target", style="cyan")
    comp_table.add_column("Flat", style="dim", justify="right")
    comp_table.add_column("BMA", style="yellow", justify="right")
    comp_table.add_column("LOO", style="green", justify="right")
    comp_table.add_column("LOO - BMA", style="magenta", justify="right")
    
    for target in targets:
        flat_arr = np.asarray(posterior_flat.get(target, []), dtype=np.float64)
        bma_arr = np.asarray(posterior_bma.get(target, []), dtype=np.float64)
        loo_arr = np.asarray(posterior_loo.get(target, []), dtype=np.float64)
        
        flat_mean = _target_mean(flat_arr)
        bma_mean = _target_mean(bma_arr)
        loo_mean = _target_mean(loo_arr)
        
        if np.asarray(flat_mean).ndim == 0:
            diff = float(loo_mean) - float(bma_mean)
            
            # Color code based on magnitude
            if abs(diff) > 0.01:
                diff_str = f"[bold magenta]{diff:+.6f}[/bold magenta]"
            elif abs(diff) > 0.001:
                diff_str = f"[magenta]{diff:+.6f}[/magenta]"
            else:
                diff_str = f"[dim]{diff:+.6f}[/dim]"
            
            comp_table.add_row(
                target,
                f"{float(flat_mean):.6f}",
                f"{float(bma_mean):.6f}",
                f"{float(loo_mean):.6f}",
                diff_str
            )
        else:
            comp_table.add_row(
                target,
                f"shape: {np.asarray(flat_mean).shape}",
                f"shape: {np.asarray(bma_mean).shape}",
                f"shape: {np.asarray(loo_mean).shape}",
                "—"
            )
    
    console.print(comp_table)
    console.print()


def _target_mean(arr):
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == 0:
        return np.nan
    return np.mean(arr, axis=0)


def _print_mean_summary(label, mean_value):
    mean_arr = np.asarray(mean_value, dtype=np.float64)
    if mean_arr.ndim == 0:
        print(f"{label} mean prediction: {float(mean_arr):.6f}")
        return
    print(f"{label} mean prediction shape: {tuple(mean_arr.shape)}")
    _print_array_preview(f"{label} mean prediction", mean_arr)


def _print_array_preview(label, arr):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    if flat.size <= 10:
        print(f"{label} full: {flat.tolist()}")
    else:
        print(f"{label} first_10: {flat[:10].tolist()}")


def _softmax_from_logs(log_values):
    shifted = log_values - np.max(log_values)
    unnorm = np.exp(shifted)
    return unnorm / np.sum(unnorm)


def _resample_weighted_samples(per_model_samples, targets, model_weights, total_draws, rng):
    out = {target: [] for target in targets}
    model_choices = rng.choice(len(per_model_samples), size=total_draws, p=model_weights)

    for m_idx in model_choices:
        samples_m = per_model_samples[m_idx]
        if any(target not in samples_m for target in targets):
            continue
        lengths = [len(samples_m[target]) for target in targets]
        min_len = int(min(lengths)) if lengths else 0
        if min_len <= 0:
            continue
        s_idx = int(rng.integers(low=0, high=min_len))
        for target in targets:
            out[target].append(samples_m[target][s_idx])

    return out


def _build_no_valid_models_message(diagnostics):
    msg = (
        "LLM produced 0 valid models out of "
        f"{diagnostics['generated_models']} generated. Cannot perform inference. "
        f"requested={diagnostics['requested_models']}, "
        f"deduplicated={diagnostics['deduplicated_models']}, "
        f"invalid_syntax_or_parsing={diagnostics['invalid_models_syntax_or_parsing']}, "
        f"generation_request_failures={diagnostics['generation_request_failures']}, "
        f"missing_targets={diagnostics['missing_targets_failures']}, "
        f"compile_failures={diagnostics['compile_failures']}, "
        f"inference_failures={diagnostics['inference_failures']}, "
        f"shape_mismatch_drops={diagnostics['shape_mismatch_drops']}, "
        f"pathological_drops={diagnostics['pathological_drops']}, "
        f"loo_failures={diagnostics['loo_failures']}, "
        f"nonfinite_log_bound_drops={diagnostics['nonfinite_log_bound_drops']}."
    )
    reason = diagnostics.get("first_failure_reason")
    if reason:
        msg += f" First failure reason: {reason}"
    return msg


def _first_request_failure_reason(gen_diag):
    failures = gen_diag.get("generation_failures", []) if isinstance(gen_diag, dict) else []
    for _slot_idx, reason in failures:
        if isinstance(reason, str) and not reason.startswith("parsing_error:"):
            return reason
    return None


def _filter_models_by_target_shape(valid_models, targets):
    if len(valid_models) <= 1:
        return valid_models, 0

    profiles = []
    for model in valid_models:
        target_samples = model.get("target_samples", {})
        try:
            draw_lengths = []
            event_shapes = []
            for target in targets:
                arr = np.asarray(target_samples[target])
                if arr.ndim < 1 or arr.shape[0] <= 0:
                    raise ValueError("empty target samples")
                draw_lengths.append(int(arr.shape[0]))
                event_shapes.append(tuple(arr.shape[1:]))
            if len(set(draw_lengths)) != 1:
                profiles.append(None)
            else:
                profiles.append(tuple(event_shapes))
        except Exception:
            profiles.append(None)

    valid_profiles = [p for p in profiles if p is not None]
    if not valid_profiles:
        return [], len(valid_models)

    counts = {}
    for p in valid_profiles:
        counts[p] = counts.get(p, 0) + 1
    dominant_profile = max(counts, key=counts.get)

    kept = []
    dropped = 0
    for model, profile in zip(valid_models, profiles):
        if profile == dominant_profile:
            kept.append(model)
        else:
            dropped += 1

    return kept, dropped


def _dedupe_model_codes(codes):
    seen = set()
    out = []
    dropped = 0
    for code in codes:
        key = _normalize_code_for_hash(code)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(code)
    return out, dropped


def _normalize_code_for_hash(code):
    lines = []
    for raw in str(code).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _normalize_target_sample_map(target_samples, targets):
    arrays = {}
    for target in targets:
        if target not in target_samples:
            return False, f"missing target '{target}'"
        arr = np.asarray(target_samples[target], dtype=np.float64)
        if arr.ndim == 0:
            return False, f"target '{target}' has scalar sample container"
        arrays[target] = arr

    first_dims = [arrays[target].shape[0] for target in targets]
    n_ref = int(max(first_dims))

    # Try to repair simple transposition errors: (event_dim, n_samples) -> (n_samples, event_dim)
    for target in targets:
        arr = arrays[target]
        if arr.shape[0] == n_ref:
            continue
        if arr.ndim >= 2 and arr.shape[1] == n_ref:
            arrays[target] = np.swapaxes(arr, 0, 1)

    first_dims_after = [arrays[target].shape[0] for target in targets]
    if len(set(first_dims_after)) != 1:
        return False, f"inconsistent sample axis lengths across targets: {first_dims_after}"

    normalized = {target: arrays[target] for target in targets}
    return True, normalized


def _resolve_report_targets(sample_maps, fallback_targets):
    if not sample_maps:
        return list(fallback_targets)

    returned = []
    seen = set()

    for target in fallback_targets:
        if all(target in sm for sm in sample_maps):
            returned.append(target)
            seen.add(target)

    for target in sample_maps[0].keys():
        if target in seen:
            continue
        if all(target in sm for sm in sample_maps):
            returned.append(target)
            seen.add(target)

    return returned