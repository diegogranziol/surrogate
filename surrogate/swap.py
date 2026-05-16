"""Swap the model behind the vLLM endpoint.

When running on the GPU box itself, uses local subprocess + skips the SSH tunnel
work. When running on a remote workstation, uses the `flow-blonde-panther` SSH
alias to drive the swap on the box. Behavior is identical either way.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import httpx


SSH_ALIAS = "flow-blonde-panther"
LOCAL_URL = "http://localhost:8000/v1/models"


def _on_box() -> bool:
    """True when this process is already running on the GPU box."""
    if os.environ.get("SURROGATE_HOST_MODE") == "box":
        return True
    # The vLLM venv lives at ~/venvs/vllm on the box and only there.
    return Path("/home/ubuntu/venvs/vllm").exists()


_LOCAL = _on_box()


@dataclass(frozen=True)
class ModelConfig:
    hf_id: str
    served_name: str
    extra_args: tuple[str, ...] = ()
    max_model_len: int = 16384

    def script(self) -> str:
        extra = " \\\n  ".join(self.extra_args)
        if extra:
            extra = "  " + extra + " \\"
        return dedent(f"""\
            #!/bin/bash
            set -e
            source ~/venvs/vllm/bin/activate
            exec python -m vllm.entrypoints.openai.api_server \\
              --model {self.hf_id} \\
              --served-model-name {self.served_name} \\
              --host 127.0.0.1 --port 8000 \\
              --max-model-len {self.max_model_len} \\
            {extra}
              2>&1 | tee -a ~/vllm.log
            """)


PRESETS: dict[str, ModelConfig] = {
    "qwen2.5-7b": ModelConfig(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        served_name="qwen2.5-7b",
        extra_args=("--enable-auto-tool-choice", "--tool-call-parser hermes"),
    ),
    "qwen3-8b": ModelConfig(
        hf_id="Qwen/Qwen3-8B",
        served_name="qwen3-8b",
        extra_args=(
            "--reasoning-parser qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser hermes",
        ),
    ),
    "qwen3-32b": ModelConfig(
        hf_id="Qwen/Qwen3-32B",
        served_name="qwen3-32b",
        # Drop max-model-len to 8192 to leave headroom for KV cache on 80GB,
        # since Qwen3-32B bf16 alone is ~64GB of weights.
        max_model_len=8192,
        extra_args=(
            "--reasoning-parser qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser hermes",
            "--gpu-memory-utilization 0.95",
        ),
    ),
}


def _run_on_box(shell_cmd: str, *, check: bool = True, capture: bool = True) -> str:
    """Run a shell command on the GPU box, whether we're already on it or not."""
    if _LOCAL:
        r = subprocess.run(["bash", "-lc", shell_cmd], capture_output=capture, text=True)
    else:
        r = subprocess.run(["ssh", SSH_ALIAS, shell_cmd], capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {shell_cmd}\n{r.stderr}")
    return (r.stdout or "") + (r.stderr or "")


def _put_file(local_path: str, remote_path: str) -> None:
    """Place a file at remote_path on the box."""
    if _LOCAL:
        # already on the box — just copy locally
        dest = Path(os.path.expanduser(remote_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local_path).read_bytes())
        return
    r = subprocess.run(["scp", local_path, f"{SSH_ALIAS}:{remote_path}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed: {r.stderr}")


def _ensure_tunnel(port: int = 8000) -> None:
    """On the Mac, ensure the SSH port-forward to the box is up. No-op on the box."""
    if _LOCAL:
        return
    try:
        httpx.get(f"http://localhost:{port}/v1/models", timeout=2.0)
        return
    except Exception:
        pass
    subprocess.run(["pkill", "-f", f"ssh -fN -L {port}"], capture_output=True)
    subprocess.run(["pkill", "-f", f"ssh -N -L {port}"], capture_output=True)
    time.sleep(1)
    subprocess.run(
        ["ssh", "-fN", "-L", f"{port}:127.0.0.1:{port}", SSH_ALIAS],
        check=True, capture_output=True,
    )
    time.sleep(1)


def current_served_name() -> str | None:
    try:
        r = httpx.get(LOCAL_URL, timeout=3.0)
        return r.json()["data"][0]["id"]
    except Exception:
        return None


def swap_to(name: str, *, log: bool = True) -> ModelConfig:
    """Ensure the model identified by `name` (a PRESETS key) is serving."""
    cfg = PRESETS[name]
    _ensure_tunnel()
    cur = current_served_name()
    if cur == cfg.served_name:
        if log:
            print(f"[swap] {cfg.served_name} already loaded")
        return cfg

    if log:
        where = "local" if _LOCAL else "remote"
        print(f"[swap/{where}] currently={cur!s} → target={cfg.served_name}")

    # Kill existing server. tmux kill-session can fail if no session — ignore.
    _run_on_box(
        "tmux kill-session -t vllm 2>/dev/null; pkill -f vllm.entrypoints; sleep 4",
        check=False,
    )

    # Write new launch script.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(cfg.script())
        tmp = f.name
    try:
        target = os.path.expanduser("~/run_vllm.sh") if _LOCAL else "~/run_vllm.sh"
        _put_file(tmp, target)
    finally:
        os.unlink(tmp)
    _run_on_box("chmod +x ~/run_vllm.sh && rm -f ~/vllm.log && tmux new-session -d -s vllm '~/run_vllm.sh'")

    # Poll readiness.
    t0 = time.time()
    deadline = t0 + 600
    while time.time() < deadline:
        out = _run_on_box("curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null || true", check=False)
        if cfg.served_name in (out or ""):
            elapsed = time.time() - t0
            if log:
                print(f"[swap] {cfg.served_name} ready ({elapsed:.0f}s)")
            break
        if log:
            tail = _run_on_box("tail -n1 ~/vllm.log 2>/dev/null | head -c 160", check=False)
            print(f"[swap] waiting… {tail.strip()[:140]}")
        time.sleep(8)
    else:
        raise TimeoutError(f"model {cfg.served_name} did not come up in 10min")

    _ensure_tunnel()
    cur = current_served_name()
    if cur != cfg.served_name:
        raise RuntimeError(f"endpoint reports {cur!r}, expected {cfg.served_name!r}")
    return cfg
