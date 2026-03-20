from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llb

text = "I have a bunch of coin flips. What's the bias?"
data = {"flips": [0, 1, 0, 1, 1, 0]}
targets = ["true_bias"]

API_KEY = 'user_open_ai_key'
API_MODEL = "gpt-4.1-mini"

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/responses",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models = 2,
    mcmc_num_warmup=5,
    mcmc_num_samples=10
)
