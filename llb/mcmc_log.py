import math

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density


def run_inference(code, data, targets=None, num_warmup=500, num_samples=1000, rng_seed=0):
    env = {}
    try:
        exec(code, env)
    except Exception as exc:
        raise ValueError(f"compile_error: {exc}") from exc

    if "model" not in env or not callable(env["model"]):
        raise ValueError("compile_error: generated code does not define callable model(data)")

    model = env["model"]

    # Reject models with unobserved discrete latent variables; these can trigger
    # automatic enumeration warnings and unstable behavior for generic NUTS usage.
    discrete_sites = _find_unobserved_discrete_sites(model=model, data=data, rng_seed=rng_seed)
    if discrete_sites:
        names = ", ".join(discrete_sites[:8])
        suffix = "" if len(discrete_sites) <= 8 else f", ... (+{len(discrete_sites) - 8} more)"
        raise ValueError(
            "inference_error: Model has unobserved discrete latent site(s) not supported by this pipeline: "
            f"{names}{suffix}. Use continuous latent variables or mark discrete structure explicitly."
        )
    
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    try:
        mcmc.run(jax.random.PRNGKey(rng_seed), data=data)
    except Exception as exc:
        msg = str(exc)
        if "TracerIntegerConversionError" in msg or "__index__() method was called on traced array" in msg:
            raise ValueError(
                "inference_error: Generated model used a traced value where a Python int is required "
                "(for example range(sampled_value) or list indexing with sampled/jnp values). "
                "Use loop bounds from static data fields instead."
            ) from exc
        raise ValueError(f"inference_error: {exc}") from exc

    samples = mcmc.get_samples(group_by_chain=False)
    available = sorted(samples.keys())

    if targets is None:
        selected_targets = available
    elif isinstance(targets, str):
        selected_targets = [targets]
    else:
        selected_targets = []
        for target in targets:
            if isinstance(target, str):
                selected_targets.append(target)
            elif isinstance(target, set):
                for item in sorted(target):
                    if not isinstance(item, str):
                        raise TypeError("target names must be strings")
                    selected_targets.append(item)
            else:
                raise TypeError("target names must be strings")

    present_targets = [name for name in selected_targets if name in samples]
    missing_targets = [name for name in selected_targets if name not in samples]
    target_samples = {name: np.asarray(samples[name]).tolist() for name in present_targets}
    return {
        "model": model,
        "samples": {name: np.asarray(value) for name, value in samples.items()},
        "target_samples": target_samples,
        "available_sites": available,
        "missing_targets": missing_targets,
    }


def estimate_log_marginal_iw(
    model,
    data,
    posterior_samples,
    num_inner=25,
    num_outer=1000,
    rng_seed=0,
    min_std=1e-4,
    fallback_log_bound=-1e12,
):
    means = {}
    stds = {}
    for name, values in posterior_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        mean, std = _finite_mean_std_axis0(arr, min_std=min_std)
        means[name] = mean
        stds[name] = std

    rng = np.random.default_rng(rng_seed)
    outer_vals = []

    for _ in range(num_outer):
        log_ws = []
        for _ in range(num_inner):
            z = {}
            log_q = 0.0
            for name in posterior_samples:
                sample = rng.normal(loc=means[name], scale=stds[name])
                z[name] = jnp.asarray(sample)

                centered = (sample - means[name]) / stds[name]
                log_q += float(
                    -0.5
                    * np.sum(
                        centered * centered
                        + np.log(2.0 * np.pi)
                        + 2.0 * np.log(stds[name])
                    )
                )

            try:
                log_joint, _ = log_density(model, (), {"data": data}, z)
                log_w = float(log_joint) - log_q
            except Exception:
                continue

            if not np.isfinite(log_w):
                continue
            log_ws.append(log_w)

        if len(log_ws) > 0:
            outer_vals.append(_logmeanexp(log_ws))

    if len(outer_vals) == 0:
        return float(fallback_log_bound)

    finite_outer = np.asarray(outer_vals, dtype=np.float64)
    finite_outer = finite_outer[np.isfinite(finite_outer)]
    if finite_outer.size == 0:
        return float(fallback_log_bound)
    return float(np.mean(finite_outer))


def _logmeanexp(values):
    vals = np.asarray(values, dtype=np.float64)
    m = vals.max()
    return float(m + math.log(np.mean(np.exp(vals - m))))


def _find_unobserved_discrete_sites(model, data, rng_seed):
    model_trace = trace(seed(model, jax.random.PRNGKey(rng_seed))).get_trace(data=data)
    names = []
    for name, site in model_trace.items():
        if site.get("type") != "sample":
            continue
        if site.get("is_observed", False):
            continue
        fn = site.get("fn")
        if bool(getattr(fn, "has_enumerate_support", False)):
            names.append(name)
    return names


def _finite_mean_std_axis0(arr, min_std=1e-4):
    if arr.ndim == 0:
        if np.isfinite(arr):
            return arr, np.asarray(min_std, dtype=np.float64)
        return np.asarray(0.0, dtype=np.float64), np.asarray(min_std, dtype=np.float64)

    finite = np.isfinite(arr)
    count = np.sum(finite, axis=0)

    safe_den = np.maximum(count, 1)
    finite_vals = np.where(finite, arr, 0.0)
    mean = np.sum(finite_vals, axis=0) / safe_den

    centered = np.where(finite, arr - mean, 0.0)
    var = np.sum(centered * centered, axis=0) / safe_den
    std = np.sqrt(var)

    no_finite = count <= 0
    mean = np.where(no_finite, 0.0, mean)
    std = np.where(no_finite, float(min_std), std)
    std = np.maximum(std, float(min_std))

    return mean, std
