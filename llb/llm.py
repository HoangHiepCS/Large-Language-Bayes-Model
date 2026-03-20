import json

import requests

class LLMClient:
    def __init__(
        self,
        api_url,
        api_key=None,
        model=None,
        provider="auto",
        extra_headers=None,
        timeout=120,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    def generate(self, prompt):
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(prompt)

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        text = self._extract_text(payload)
        if not isinstance(text, str) or not text.strip():
            keys = sorted(payload.keys()) if isinstance(payload, dict) else [type(payload).__name__]
            raise RuntimeError(
                "Unable to parse text from LLM response. "
                f"provider={self._resolved_provider()} api_url={self.api_url} payload_keys={keys}"
            )
        return text

    def _build_payload(self, prompt):
        mode = self._resolved_provider()

        if mode == "openai_responses":
            return {
                "model": self.model or "gpt-4.1-mini",
                "input": prompt,
            }

        if mode == "openai_chat":
            return {
                "model": self.model or "gpt-4.1-mini",
                "messages": [{"role": "user", "content": prompt}],
            }

        if mode == "ollama_generate":
            payload = {
                "prompt": prompt,
                "stream": False,
            }
            if self.model:
                payload["model"] = self.model
            return payload

        payload = {"prompt": prompt}
        if self.model:
            payload["model"] = self.model
        return payload

    def _resolved_provider(self):
        if self.provider != "auto":
            return self.provider

        url = self.api_url.lower()
        if "/v1/responses" in url:
            return "openai_responses"
        if "/v1/chat/completions" in url:
            return "openai_chat"
        if "/api/generate" in url:
            return "ollama_generate"
        return "generic_prompt"

    def _extract_text(self, payload):
        if not isinstance(payload, dict):
            return None

        # OpenAI responses API
        if payload.get("output_text"):
            return payload["output_text"]

        if "output" in payload:
            texts = []
            for item in payload["output"]:
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        texts.append(content["text"])
            if texts:
                return "\n".join(texts)

        if "choices" in payload and payload["choices"]:
            msg = payload["choices"][0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and item.get("text"):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)

        if payload.get("response"):
            return payload["response"]
        if payload.get("text"):
            return payload["text"]

        if payload.get("data") and isinstance(payload["data"], str):
            try:
                nested = json.loads(payload["data"])
                return self._extract_text(nested)
            except json.JSONDecodeError:
                return None

        return None
