from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llb

text = (
    "I've been recording if it rains each day, with a 1 for rain and 0 for no rain. "
    "Maybe there's some kind of pattern? Predict if it will rain the next day."
)

rain = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
data = {
    "num_days": len(rain),
    "rain": rain,
}

targets = ["outcome_for_next_day"]

API_KEY = 'user_open_ai_key'
API_MODEL = "gpt-4.1-mini"

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/responses",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models=16,
    llm_timeout=600,
    mcmc_num_warmup=500,
    mcmc_num_samples=1000,
)
