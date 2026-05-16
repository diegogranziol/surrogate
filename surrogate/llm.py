from __future__ import annotations

import os

from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


BASE_URL = os.environ.get("SURROGATE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("SURROGATE_MODEL", "qwen2.5-7b")


def make_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=os.environ.get("SURROGATE_API_KEY", "EMPTY"))
