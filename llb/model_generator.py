import re

from .examples import NUMPYRO_EXAMPLES


def build_prompt(text, data, targets):
    goal = targets if targets is not None else []
    return f"""You are a probabilistic programmer writing a NumPyro Bayesian model.

CRITICAL REQUIREMENTS:
1. Write valid NumPyro code that compiles and runs without errors
2. Define exactly ONE function called `def model(data):`
3. Use vectorized operations for multiple observations (numpy/jax arrays)
4. Each observation sample site must have a UNIQUE NAME
5. Include all necessary imports (numpyro, jax, distributions)
6. For each goal variable name, return a deterministic output with exactly that name using `numpyro.deterministic(name, value)`
7. Do NOT sample goal variable names as latent sites (no `numpyro.sample('<goal>', ...)` unless it is an observed site with `obs=`)
8. Every NumPyro site name must be globally unique across `sample`, `deterministic`, and `plate` names
9. Goal names are RESERVED: if a goal is `g`, only `numpyro.deterministic('g', ...)` may use that exact name

IMPORTANT PATTERNS:
- Use `numpyro.plate()` to vectorize over observations
- Use `numpyro.sample('var_name', dist.SomeDistribution(), obs=data)` for observations
- Each `numpyro.sample()` call needs a unique name
- Do NOT reuse any deterministic site name as a sample site name
- Do NOT create multiple observation sites with the same name
- Goal variables should be probabilities or continuous summaries exposed with `numpyro.deterministic`
- If a goal is boolean-like (e.g., true/next/yes), expose probability of True as deterministic value in [0, 1]

EXAMPLE (DO THIS):
```python
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    p = numpyro.sample('p', dist.Beta(1, 1))
    with numpyro.plate('obs_plate', len(data['observations'])):
        numpyro.sample('obs', dist.Bernoulli(p), obs=jnp.array(data['observations']))
```

IN-CONTEXT EXAMPLES (translated from paper Stan models):
{NUMPYRO_EXAMPLES}

PROBLEM:
{text}

DATA:
{data}

GOAL: Generate latent variables: {goal}

IMPORTANT GOAL FORMAT:
- Every goal name in {goal} must appear exactly as a `numpyro.deterministic('<goal>', ...)` site.
- Do not generate `numpyro.sample('<goal>', dist.Bernoulli(...))` or `numpyro.sample('<goal>', dist.Categorical(...))` for goal names.
- Never use goal names for any other site type: no `sample`, no `plate`, and no duplicate deterministic definitions.

FINAL SELF-CHECK BEFORE OUTPUT:
- Confirm all site names are unique.
- Confirm each goal appears exactly once and only as deterministic.
- Confirm no goal name appears in any `numpyro.sample(...)` statement.

Now write the complete model function inside a ```python code block:"""


def extract_model_code(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(
            "LLM returned empty or non-text output; check API URL/provider format and response parsing."
        )

    text = raw_text.strip()

    block_matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in block_matches:
        candidate = block.strip()
        if "def model(" in candidate:
            code = candidate
            # Ensure imports are present
            return _add_imports_if_needed(code)

    idx = text.find("def model(")
    if idx != -1:
        code = text[idx:].strip()
        return _add_imports_if_needed(code)

    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith('def ') and 'model' in stripped:
            code = '\n'.join(lines[i:]).strip()
            return _add_imports_if_needed(code)

    return _add_imports_if_needed(text)


def _add_imports_if_needed(code):
    """Add required imports if they're not already present."""
    lines = code.split('\n')
    
    # Check what's already imported
    has_numpyro = any('import numpyro' in line for line in lines[:20])
    has_dist = any('numpyro.distributions' in line or 'import dist' in line for line in lines[:20])
    has_jnp = any('import jax.numpy as jnp' in line for line in lines[:30])
    has_np = any('import numpy as np' in line for line in lines[:30])

    uses_jnp = re.search(r'\bjnp\s*\.', code) is not None
    uses_np = re.search(r'\bnp\s*\.', code) is not None
    
    imports_needed = []
    if not has_numpyro:
        imports_needed.append("import numpyro")
    if not has_dist:
        imports_needed.append("import numpyro.distributions as dist")
    if uses_jnp and not has_jnp:
        imports_needed.append("import jax.numpy as jnp")
    if uses_np and not has_np:
        imports_needed.append("import numpy as np")
    
    if imports_needed:
        # Find where to insert imports (before the first def or at the start)
        def_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), 0)
        for imp in reversed(imports_needed):
            lines.insert(def_idx, imp)
    
    return '\n'.join(lines)


def generate_models(llm, text, data, targets, n_models):
    models = []
    for _ in range(n_models):
        prompt = build_prompt(text, data, targets)
        code = extract_model_code(llm.generate(prompt))
        models.append(code)
    return models
