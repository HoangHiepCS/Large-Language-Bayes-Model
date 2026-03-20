from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llb

text = (
    "I have a coin. I've flipped it a bunch of times. I'm wondering what the true bias of the coin is. "
    "I just got the coin from the US mint, so I'm almost completely sure that it's a standard US penny."
)
data = {"num_flips": 20, "num_heads": 14}
targets = ["bias"]

API_KEY = 'user_open_ai_key'
API_MODEL = "gpt-4.1-mini"

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/responses",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models=8,
    mcmc_num_warmup=500,
    mcmc_num_samples=1000,
)

print(f"posterior draws for bias: {len(posterior['bias'])}")
