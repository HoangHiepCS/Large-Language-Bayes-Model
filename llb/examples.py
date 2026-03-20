NUMPYRO_EXAMPLES = """
Example 1:
PROBLEM:
Predict next-day rain from a binary rain time series.
DATA:
{"rain": [1, 0, 1, 1, 0, 0, 1]}
GOAL:
["next"]
MODEL:
def model(data):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    rain = jnp.array(data["rain"])
    num_days = rain.shape[0]

    logit_initial = numpyro.sample("logit_initial", dist.Uniform(-10.0, 10.0))
    autoregressive_coefficient = numpyro.sample("autoregressive_coefficient", dist.Normal(0.0, 1.0))

    logits = [logit_initial]
    for i in range(1, int(num_days)):
        noise_i = numpyro.sample(f"noise_{i}", dist.Cauchy(0.0, 10.0))
        logits.append(logits[-1] * autoregressive_coefficient + noise_i)

    for i in range(int(num_days)):
        numpyro.sample(f"obs_{i}", dist.Bernoulli(logits=logits[i]), obs=rain[i])

    logit_next = logits[-1] * autoregressive_coefficient + numpyro.sample("next_noise", dist.Cauchy(0.0, 10.0))
    numpyro.sample("next", dist.Bernoulli(logits=logit_next))

Example 2):
PROBLEM:
Infer coin bias with a strong prior around a standard coin.
DATA:
{"num_flips": 20, "num_heads": 14}
GOAL:
["bias"]
MODEL:
def model(data):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    logit_bias = numpyro.sample("logit_bias", dist.Normal(0.0, 0.1))
    bias = numpyro.deterministic("bias", jnp.clip(jnp.exp(logit_bias) / (1.0 + jnp.exp(logit_bias)), 1e-6, 1.0 - 1e-6))
    numpyro.sample("obs", dist.Binomial(total_count=data["num_flips"], probs=bias), obs=data["num_heads"])

Example 3:
PROBLEM:
Infer coin bias with weak prior knowledge.
DATA:
{"num_flips": 20, "num_heads": 14}
GOAL:
["bias"]
MODEL:
def model(data):
    import numpyro
    import numpyro.distributions as dist

    bias = numpyro.sample("bias", dist.Uniform(0.0, 1.0))
    numpyro.sample("obs", dist.Binomial(total_count=data["num_flips"], probs=bias), obs=data["num_heads"])

Example 4:
PROBLEM:
Infer coin bias when the coin may be physically bent.
DATA:
{"num_flips": 20, "num_heads": 14}
GOAL:
["bias"]
MODEL:
def model(data):
    import numpyro
    import numpyro.distributions as dist

    bias = numpyro.sample("bias", dist.Beta(0.5, 0.5))
    numpyro.sample("obs", dist.Binomial(total_count=data["num_flips"], probs=bias), obs=data["num_heads"])

Example 5:
PROBLEM:
Infer daily popularity from noisy multi-pollster observations.
DATA:
{"num_days": 5, "num_pollsters": 2, "num_polls": [3, 2], "day": [[1, 3, 5], [2, 4, 0]], "polls": [[48.0, 50.0, 52.0], [49.0, 51.0, 0.0]]}
GOAL:
["popularity_1", "popularity_2", "popularity_3", "popularity_4", "popularity_5"]
MODEL:
def model(data):
    import numpyro
    import numpyro.distributions as dist

    num_days = int(data["num_days"])
    num_pollsters = int(data["num_pollsters"])

    popularity = [numpyro.sample("popularity_1", dist.Uniform(0.0, 100.0))]
    for t in range(2, num_days + 1):
        popularity_t = numpyro.sample(f"popularity_{t}", dist.Normal(popularity[-1], 10.0))
        popularity.append(popularity_t)

    for i in range(num_pollsters):
        sigma_i = numpyro.sample(f"sigma_{i+1}", dist.InverseGamma(5.0, 20.0))
        n_i = int(data["num_polls"][i])
        for j in range(n_i):
            day_ij = int(data["day"][i][j])
            poll_ij = float(data["polls"][i][j])
            numpyro.sample(f"obs_{i+1}_{j+1}", dist.Normal(popularity[day_ij - 1], sigma_i), obs=poll_ij)

Example 6:
PROBLEM:
Predict next-day city temperatures from prior-day temperatures.
DATA:
{"num_cities": 2, "num_days_train": 3, "num_days_test": 2, "day1_temp_train": [[60.0, 62.0, 58.0], [70.0, 68.0, 72.0]], "day2_temp_train": [[61.0, 63.0, 57.5], [69.0, 67.5, 73.0]], "day1_temp_test": [[59.0, 64.0], [71.0, 69.0]]}
GOAL:
["beta", "alpha"]
MODEL:
def model(data):
    import numpyro
    import numpyro.distributions as dist

    num_cities = int(data["num_cities"])
    num_days_train = int(data["num_days_train"])

    beta = numpyro.sample("beta", dist.Normal(0.8, 0.2))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))

    for c in range(num_cities):
        city_mean = numpyro.sample(f"city_mean_{c+1}", dist.Normal(50.0, 20.0))
        for d in range(num_days_train):
            day1_adj = numpyro.sample(f"day1_temp_adj_{c+1}_{d+1}", dist.Normal(0.0, 10.0))
            mean_cd = city_mean + alpha + beta * (data["day1_temp_train"][c][d] - city_mean + day1_adj)
            numpyro.sample(
                f"obs_{c+1}_{d+1}",
                dist.Normal(mean_cd, 5.0),
                obs=data["day2_temp_train"][c][d],
            )

Example 7:
PROBLEM:
Predict binary gold presence along a rod from observed locations.
DATA:
{"rod_length": 100.0, "num_train": 5, "train_locs": [10.0, 25.0, 40.0, 60.0, 85.0], "gold_train": [0, 1, 0, 1, 1], "num_test": 3, "test_locs": [15.0, 50.0, 90.0]}
GOAL:
["gold_test_1", "gold_test_2", "gold_test_3"]
MODEL:
def model(data):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    train_locs = jnp.array(data["train_locs"])
    test_locs = jnp.array(data["test_locs"])
    all_locs = jnp.concatenate([train_locs, test_locs])
    n = all_locs.shape[0]

    rho = numpyro.sample("rho", dist.HalfNormal(data["rod_length"]))
    eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([n]).to_event(1))

    loc_diffs = all_locs[:, None] - all_locs[None, :]
    dists = jnp.sqrt(loc_diffs * loc_diffs + 1e-8)
    K = jnp.eye(n) * 1.01 + jnp.sqrt(5.0 / (3.0 * rho + 1e-6)) * jnp.power(1.0 + jnp.sqrt(5.0 * dists / (rho + 1e-6)), -2.0) * jnp.exp(-jnp.sqrt(5.0 * dists / (3.0 * rho + 1e-6)) * 1.2)
    f = K @ eta

    num_train = int(data["num_train"])
    for i in range(num_train):
        numpyro.sample(f"obs_{i+1}", dist.Bernoulli(logits=f[i]), obs=data["gold_train"][i])

    num_test = int(data["num_test"])
    for i in range(num_test):
        numpyro.sample(f"gold_test_{i+1}", dist.Bernoulli(logits=f[num_train + i]))

Example 8:
PROBLEM:
Large-data version of the gold-location prediction problem.
DATA:
{"rod_length": 100.0, "num_train": 6, "train_locs": [8.0, 19.0, 31.0, 47.0, 73.0, 94.0], "gold_train": [0, 0, 1, 0, 1, 1], "num_test": 2, "test_locs": [55.0, 88.0]}
GOAL:
["gold_test_1", "gold_test_2"]
MODEL:
def model(data):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    train_locs = jnp.array(data["train_locs"])
    test_locs = jnp.array(data["test_locs"])
    all_locs = jnp.concatenate([train_locs, test_locs])
    n = all_locs.shape[0]

    eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    rho = numpyro.sample("rho", dist.HalfNormal(data["rod_length"]))

    loc_diffs = all_locs[:, None] - all_locs[None, :]
    dists = jnp.sqrt(loc_diffs * loc_diffs + 1e-8)
    K = jnp.eye(n) * 1.01 + jnp.sqrt(5.0 / (3.0 * rho + 1e-6)) * jnp.power(1.0 + jnp.sqrt(5.0 * dists / (rho + 1e-6)), -2.0) * jnp.exp(-jnp.sqrt(5.0 * dists / (3.0 * rho + 1e-6)) * 1.2)
    f = K @ eta

    num_train = int(data["num_train"])
    for i in range(num_train):
        numpyro.sample(f"obs_{i+1}", dist.Bernoulli(logits=f[i]), obs=data["gold_train"][i])

    num_test = int(data["num_test"])
    for i in range(num_test):
        numpyro.sample(f"gold_test_{i+1}", dist.Bernoulli(logits=f[num_train + i]))
"""
