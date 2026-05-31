"""Box (remote-GPU) configuration + tunnel control.

Persists the current SSH-tunnel target to `box_config.json` (gitignored)
so the Streamlit Settings tab can show "last used" values and switch boxes
without anyone editing scripts.

The bash tunnel keeper at `scripts/keep_tunnel.sh` already reads
TUNNEL_HOST / TUNNEL_USER / TUNNEL_PORT / TUNNEL_KEY / TUNNEL_LOCAL_PORT
from the environment. We start it with that env populated from the JSON.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx

SETTINGS_PATH = Path("box_config.json")
USER_PRESETS_PATH = Path("box_presets.json")
KEEPER_SCRIPT = Path("scripts/keep_tunnel.sh")
KEEPER_LOG = Path("/tmp/keep_tunnel.log")

# Known presets — clicking one in the UI fills the form fields. The user
# can still hand-edit before saving.
_BASE = {
    "reference_model": "glm-4.6",   # GLM model used by surrogate.reference
}

PRESETS: dict[str, dict] = {
    "Mithril (B200) — current": {
        **_BASE,
        "host": "44.250.249.199",
        "user": "ubuntu",
        "port": 22,
        "key": str(Path.home() / "ulusha-key.pem"),
        "local_port": 8000,
    },
    "vast.ai (legacy 3090)": {
        **_BASE,
        "host": "81.166.173.12",
        "user": "root",
        "port": 10753,
        "key": "",          # empty -> ssh-config / agent
        "local_port": 8000,
    },
}

# Common reference-model names; the UI text input accepts anything.
REFERENCE_MODEL_SUGGESTIONS = ["glm-4.6", "glm-5.1", "glm-4.5-air", "glm-4.5"]

DEFAULTS = PRESETS["Mithril (B200) — current"].copy()


def load_user_presets() -> dict[str, dict]:
    """User-added presets, persisted to USER_PRESETS_PATH (gitignored).
    Returns {} if the file is missing or unreadable."""
    if not USER_PRESETS_PATH.exists():
        return {}
    try:
        data = json.loads(USER_PRESETS_PATH.read_text())
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        print(f"[box] failed to read {USER_PRESETS_PATH}: {e!r}")
    return {}


def save_user_preset(name: str, settings: dict) -> None:
    """Add or update a user preset. Only stores the canonical preset keys —
    strips last_used and any other extras."""
    presets = load_user_presets()
    clean = {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS}
    presets[name] = clean
    USER_PRESETS_PATH.write_text(json.dumps(presets, indent=2) + "\n")


def delete_user_preset(name: str) -> bool:
    """Remove a user preset. Factory presets are read-only; deletion of a
    factory preset name is silently refused."""
    if name in PRESETS:
        return False
    presets = load_user_presets()
    if name not in presets:
        return False
    del presets[name]
    USER_PRESETS_PATH.write_text(json.dumps(presets, indent=2) + "\n")
    return True


def is_factory_preset(name: str) -> bool:
    return name in PRESETS


def all_presets() -> dict[str, dict]:
    """Factory presets first, then user presets. Users can shadow factory
    names if they really want to."""
    out = dict(PRESETS)
    out.update(load_user_presets())
    return out


def load_settings() -> dict:
    """Return the persisted settings if present, otherwise the Mithril preset."""
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            # Fill any missing keys with defaults so the UI never crashes.
            out = DEFAULTS.copy()
            out.update({k: v for k, v in data.items() if k in DEFAULTS or k == "last_used"})
            return out
        except Exception as e:
            print(f"[box] failed to read {SETTINGS_PATH}: {e!r}; using defaults")
    return DEFAULTS.copy()


def save_settings(settings: dict) -> dict:
    """Persist the settings, stamping `last_used`, and push the values that
    are env-driven (REFERENCE_MODEL) into os.environ so the running process
    picks them up without a restart."""
    out = {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS}
    out["last_used"] = datetime.now().isoformat(timespec="seconds")
    SETTINGS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    apply_to_env(out)
    return out


def apply_to_env(settings: dict | None = None) -> None:
    """Push env-driven settings into os.environ so live code sees them.

    Call this at app startup (after load_settings) and after save_settings.
    """
    if settings is None:
        settings = load_settings()
    rm = (settings.get("reference_model") or "").strip()
    if rm:
        os.environ["REFERENCE_MODEL"] = rm


# --- tunnel keeper control --------------------------------------------------

def _running_keeper_pids() -> list[int]:
    """PIDs of any keep_tunnel.sh processes currently running on this Mac."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "keep_tunnel.sh"],
            capture_output=True, text=True, check=False,
        )
        return [int(p) for p in r.stdout.split() if p.strip().isdigit()]
    except Exception:
        return []


def stop_tunnel() -> int:
    """Kill any running keeper + child ssh. Returns how many were killed."""
    killed = 0
    for pid in _running_keeper_pids():
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except Exception:
            pass
    # Also kill orphaned `ssh -fN -L 8000:127.0.0.1:8000` (any local_port)
    subprocess.run(
        ["pkill", "-f", "ssh.*-L.*:127.0.0.1:8000"],
        capture_output=True, check=False,
    )
    return killed


def start_tunnel(settings: dict | None = None) -> dict:
    """Spawn the keep_tunnel.sh detached, with env populated from `settings`
    (or the persisted JSON if `settings` is None). Returns a status dict.
    """
    if settings is None:
        settings = load_settings()
    if not KEEPER_SCRIPT.exists():
        return {"ok": False, "error": f"keeper script not found at {KEEPER_SCRIPT}"}
    if not settings.get("host"):
        return {"ok": False, "error": "no host configured"}

    # Make sure the script is executable.
    try:
        KEEPER_SCRIPT.chmod(KEEPER_SCRIPT.stat().st_mode | 0o111)
    except Exception:
        pass

    env = os.environ.copy()
    env["TUNNEL_HOST"] = str(settings["host"])
    env["TUNNEL_USER"] = str(settings.get("user") or "ubuntu")
    env["TUNNEL_PORT"] = str(settings.get("port") or 22)
    env["TUNNEL_KEY"] = str(settings.get("key") or "")
    env["TUNNEL_LOCAL_PORT"] = str(settings.get("local_port") or 8000)

    # Use start_new_session=True (setsid) so the keeper survives the parent
    # Streamlit / Python process. Redirect stdio to the keeper log.
    log_fh = open(KEEPER_LOG, "a")
    try:
        proc = subprocess.Popen(
            [str(KEEPER_SCRIPT)],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path.cwd()),
        )
    except Exception as e:
        return {"ok": False, "error": f"failed to spawn keeper: {e!r}"}
    return {"ok": True, "pid": proc.pid, "log": str(KEEPER_LOG)}


def restart_tunnel(settings: dict | None = None) -> dict:
    """Stop any running keeper, then start with the given settings."""
    n = stop_tunnel()
    time.sleep(1.0)
    res = start_tunnel(settings)
    res["killed_prior"] = n
    return res


# --- remote SSH helpers (build the command, run it, parse the result) -------

def _ssh_base(settings: dict, *, connect_timeout: int = 15) -> list[str]:
    cmd = ["ssh"]
    key = (settings.get("key") or "").strip()
    if key:
        cmd += ["-i", key]
    cmd += [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/tmp/_box_probe_hosts",
        "-o", f"ConnectTimeout={connect_timeout}",
        "-p", str(settings.get("port") or 22),
        f"{settings.get('user') or 'ubuntu'}@{settings['host']}",
    ]
    return cmd


def _run_remote(settings: dict, remote_cmd: str, *, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run one remote shell command. Returns CompletedProcess (no exception
    on non-zero; caller inspects stdout/stderr/returncode)."""
    return subprocess.run(
        _ssh_base(settings) + [remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _run_remote_streaming(
    settings: dict,
    remote_cmd: str,
    *,
    on_progress,
    stage: str,
    heartbeat_label: str,
    timeout: float = 900.0,
    heartbeat_interval: float = 4.0,
    interesting_substrings: tuple = (
        "downloading", "installing collected", "successfully installed",
        "collecting", "building wheel", "preparing metadata",
    ),
) -> tuple[int, str]:
    """Run a long remote command via Popen, stream lines as they arrive, and
    emit progress heartbeats every `heartbeat_interval` seconds so a slow
    operation never looks frozen in the UI.

    Returns (returncode, combined_output). on_progress(stage, msg, ok) is
    called as lines come in and on each heartbeat.
    """
    import select  # POSIX (Mac/Linux) — fine for our setting.
    cmd = _ssh_base(settings) + [remote_cmd]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered (best effort over a pipe)
    )
    start = time.time()
    last_emit = start
    captured: list[str] = []
    eof = False
    try:
        # Read while process is alive OR there's still buffered output.
        while not eof:
            still_running = proc.poll() is None
            rl, _, _ = select.select(
                [proc.stdout] if proc.stdout else [], [], [], 1.0
            )
            now = time.time()
            if rl:
                line = proc.stdout.readline()
                if line == "":
                    # readable() said yes but readline got empty -> EOF.
                    eof = True
                else:
                    captured.append(line)
                    low = line.lower()
                    if any(s in low for s in interesting_substrings):
                        on_progress(stage, line.strip()[:160])
                        last_emit = now
            elif not still_running:
                # No data to read AND process has exited -> done.
                eof = True
            # Heartbeat — fires when no interesting line has shown for a while.
            if not eof and now - last_emit > heartbeat_interval:
                on_progress(stage, f"{heartbeat_label} · {int(now - start)}s elapsed")
                last_emit = now
            if (now - start) > timeout:
                proc.kill()
                on_progress(stage,
                            f"⚠️ timed out after {int(timeout)}s — killed", ok=False)
                break
    finally:
        try:
            rest = proc.stdout.read() if proc.stdout else ""
        except Exception:
            rest = ""
        if rest:
            captured.append(rest)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    return proc.returncode, "".join(captured)


# --- the actual end-to-end "make this box ready" flow -----------------------

VLLM_LAUNCHER_REMOTE_PATH = "/tmp/surrogate_launch_vllm.sh"
VLLM_LOG_REMOTE_PATH = "/tmp/surrogate_vllm.log"


def _build_launcher(model_id: str, served_name: str, max_model_len: int = 16384) -> str:
    return (
        "#!/bin/bash\n"
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        f"exec python3 -m vllm.entrypoints.openai.api_server \\\n"
        f"  --model {model_id} \\\n"
        f"  --served-model-name {served_name} \\\n"
        "  --host 127.0.0.1 --port 8000 \\\n"
        f"  --max-model-len {max_model_len} \\\n"
        "  --reasoning-parser qwen3 \\\n"
        "  --enable-auto-tool-choice \\\n"
        "  --tool-call-parser hermes\n"
    )


def provision_remote(
    settings: dict,
    *,
    model_id: str = "Qwen/Qwen3-8B",
    served_name: str = "qwen3-8b",
    on_progress=None,
    install_timeout: int = 900,
    ready_timeout: int = 900,
) -> dict:
    """Idempotently bring up vLLM serving `model_id` on the remote box.

    Detects what's already there (vLLM installed? ninja on PATH? vLLM running
    + serving the right model?) and only does the work that's missing.
    Calls `on_progress(stage, message, ok=True)` after each step.
    """
    def emit(stage, msg, ok=True):
        if on_progress:
            try:
                on_progress(stage, msg, ok)
            except Exception:
                pass

    if not (settings.get("host") or "").strip():
        return {"ok": False, "stage": "input", "error": "no host configured"}

    # 1. SSH reachable?
    emit("ssh", "connecting…")
    try:
        r = _run_remote(settings, "echo __SSH_OK__", timeout=20)
    except subprocess.TimeoutExpired:
        emit("ssh", "timeout", ok=False)
        return {"ok": False, "stage": "ssh", "error": "timeout"}
    if "__SSH_OK__" not in (r.stdout or ""):
        err = (r.stderr or r.stdout or "").strip()[-300:]
        emit("ssh", f"failed — {err}", ok=False)
        return {"ok": False, "stage": "ssh", "error": err}
    emit("ssh", "reachable")

    # 2. Inspect the box: GPU / disk / vllm / ninja / running process
    emit("inspect", "checking…")
    probe = _run_remote(settings, (
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null "
        "  | head -1 || echo NO_GPU;\n"
        "df -h / | tail -1 | awk '{print $4}';\n"
        "(python3 -c 'import vllm; print(\"VLLM_OK:\"+vllm.__version__)' 2>/dev/null "
        "  || echo VLLM_MISSING);\n"
        "(command -v ninja >/dev/null && echo NINJA_OK) "
        "  || (test -x \"$HOME/.local/bin/ninja\" && echo NINJA_USER) "
        "  || echo NINJA_MISSING;\n"
        # Scope to python processes only — otherwise the ssh shell running
        # this probe self-matches (its cmdline contains 'vllm.entrypoints').
        "(pgrep -f 'python.*vllm.entrypoints' >/dev/null && echo VLLM_RUNNING) "
        "  || echo VLLM_NOT_RUNNING;\n"
        "(curl -sf -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null "
        f"  | grep -q {served_name} && echo SERVING_OK) || echo SERVING_NO"
    ), timeout=40)
    lines = (probe.stdout or "").strip().splitlines()
    gpu = lines[0] if len(lines) > 0 else "?"
    disk_free = lines[1] if len(lines) > 1 else "?"
    vllm_state = lines[2] if len(lines) > 2 else "VLLM_MISSING"
    ninja_state = lines[3] if len(lines) > 3 else "NINJA_MISSING"
    running_state = lines[4] if len(lines) > 4 else "VLLM_NOT_RUNNING"
    serving_state = lines[5] if len(lines) > 5 else "SERVING_NO"

    # Tidy the raw probe tokens into short labels for display.
    gpu_short = (gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                 .replace(" MiB", "M").strip())
    vllm_short = ("missing" if vllm_state == "VLLM_MISSING"
                  else vllm_state.split(":", 1)[-1])
    ninja_short = {"NINJA_OK": "ok",
                   "NINJA_USER": "ok (user)",
                   "NINJA_MISSING": "missing"}.get(ninja_state, ninja_state.lower())
    emit("inspect", f"{gpu_short} · disk {disk_free} free · "
                    f"vllm: {vllm_short} · ninja: {ninja_short}")

    if "NO_GPU" in gpu:
        emit("inspect", "no GPU on this box", ok=False)
        return {"ok": False, "stage": "inspect", "error": "no GPU"}

    # FAST PATH: already serving the right model → just verify and return.
    if serving_state == "SERVING_OK":
        emit("done", f"already serving {served_name}")
        return {"ok": True, "stage": "already_serving", "model": served_name,
                "details": {"gpu": gpu, "disk_free": disk_free}}

    # 3. Install ninja if needed (vLLM's flashinfer JIT needs it on B-class GPUs)
    if ninja_state == "NINJA_MISSING":
        emit("ninja", "installing…")
        _run_remote(settings, (
            "sudo -n apt-get install -y -qq ninja-build 2>&1 | tail -2 || "
            "python3 -m pip install --user --break-system-packages --quiet ninja"
        ), timeout=180)
        r2 = _run_remote(settings, (
            "(command -v ninja >/dev/null && echo PATH) || "
            "(test -x \"$HOME/.local/bin/ninja\" && echo USER) || echo MISSING"
        ), timeout=15)
        if "MISSING" in (r2.stdout or ""):
            emit("ninja", "install failed; continuing", ok=False)
        else:
            emit("ninja", "installed")
    else:
        emit("ninja", "ok")

    # 4. Install vLLM if needed (LONG — stream lines + heartbeat).
    if vllm_state == "VLLM_MISSING":
        emit("vllm", "installing… (3–6 min)")
        rc, out = _run_remote_streaming(
            settings,
            "python3 -u -m pip install --user --break-system-packages "
            "--progress-bar off 'vllm>=0.6.0' 2>&1",
            on_progress=on_progress,
            stage="vllm",
            heartbeat_label="installing…",
            timeout=install_timeout,
            heartbeat_interval=5.0,
        )
        r3v = _run_remote(settings,
                          "python3 -c 'import vllm; print(vllm.__version__)' 2>&1",
                          timeout=30)
        ok = (rc == 0 and bool(r3v.stdout.strip())
              and "Error" not in r3v.stdout and "Module" not in r3v.stdout)
        if not ok:
            tail = (out or r3v.stdout)[-400:]
            emit("vllm", f"install failed — {tail[-160:]}", ok=False)
            return {"ok": False, "stage": "vllm-install", "error": tail}
        emit("vllm", f"installed ({r3v.stdout.strip()})")
    else:
        emit("vllm", f"ok ({vllm_state.split(':',1)[-1]})")

    # 5. Kill any existing vLLM (it may be on the wrong model or stuck)
    if running_state == "VLLM_RUNNING":
        emit("kill", "stopping old vllm…")
        _run_remote(settings, "pkill -9 -f 'python.*vllm.entrypoints' 2>/dev/null; sleep 1; echo done",
                    timeout=15)
        emit("kill", "stopped")

    # 6. Write the launcher script and start it detached
    emit("launch", f"spawning… ({model_id})")
    launcher = _build_launcher(model_id, served_name)
    write_cmd = (
        f"cat > {VLLM_LAUNCHER_REMOTE_PATH}; chmod +x {VLLM_LAUNCHER_REMOTE_PATH}; "
        f"setsid nohup {VLLM_LAUNCHER_REMOTE_PATH} </dev/null "
        f"> {VLLM_LOG_REMOTE_PATH} 2>&1 & disown 2>/dev/null; sleep 2; "
        "ps -ef | grep vllm.entrypoints | grep -v grep | head -1 || echo NOT_STARTED"
    )
    r4 = subprocess.run(
        _ssh_base(settings) + [write_cmd],
        input=launcher, capture_output=True, text=True, timeout=60,
    )
    if "vllm.entrypoints" not in (r4.stdout or ""):
        emit("launch", f"failed — {(r4.stderr or r4.stdout)[-160:]}", ok=False)
        return {"ok": False, "stage": "launch",
                "error": (r4.stderr or r4.stdout)[-300:]}
    emit("launch", "spawned")

    # 7. Poll until the box is serving the right model
    emit("ready", "loading weights… (typically 1–5 min)")
    deadline = time.time() + ready_timeout
    start_t = time.time()
    last_msg = ""
    while time.time() < deadline:
        try:
            rp = _run_remote(settings, (
                f"if curl -sf -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q {served_name}; then echo SERVING; "
                f"elif grep -qiE 'No space left|CUDA out of memory|Traceback|Killed|RuntimeError' {VLLM_LOG_REMOTE_PATH} 2>/dev/null; then echo ERR; tail -10 {VLLM_LOG_REMOTE_PATH}; "
                f"else echo BOOT; tail -1 {VLLM_LOG_REMOTE_PATH} 2>/dev/null | head -c 200; fi"
            ), timeout=25)
        except subprocess.TimeoutExpired:
            emit("ready", "ssh timeout, retrying…")
            time.sleep(5); continue
        out = (rp.stdout or "").strip()
        if out.startswith("SERVING"):
            emit("ready", f"serving {served_name}")
            return {"ok": True, "stage": "ready", "model": served_name,
                    "details": {"gpu": gpu, "disk_free": disk_free}}
        if out.startswith("ERR"):
            emit("ready", f"vllm error — {out[3:].strip()[:200]}", ok=False)
            return {"ok": False, "stage": "ready", "error": out[3:].strip()}
        msg = out[len("BOOT"):].strip() if out.startswith("BOOT") else out
        elapsed = int(time.time() - start_t)
        if msg and msg != last_msg:
            emit("ready", f"loading… {elapsed}s · {msg[:120]}")
            last_msg = msg
        else:
            emit("ready", f"loading… {elapsed}s")
        time.sleep(8)

    emit("ready", "timeout waiting for serve", ok=False)
    return {"ok": False, "stage": "ready", "error": "timeout"}


# --- endpoint probe ---------------------------------------------------------

def is_endpoint_alive(local_port: int = 8000, timeout: float = 4.0) -> dict:
    """Probe http://localhost:<local_port>/v1/models — reflects the CURRENT
    tunnel state (which was started from the SAVED settings). Used by the
    status badges at the top of the Settings tab."""
    url = f"http://localhost:{local_port}/v1/models"
    try:
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        model = (data.get("data") or [{}])[0].get("id") if data.get("data") else None
        return {"ok": True, "model": model, "error": None}
    except Exception as e:
        return {"ok": False, "model": None, "error": repr(e)}


def probe_remote(settings: dict, timeout: float = 12.0) -> dict:
    """Test the values the user just TYPED — no tunnel, no save. SSH straight
    to the box with those settings and check if vLLM is listening on its
    local port 8000. Lets the user validate a config BEFORE persisting it.

    Returns:
        {
          'ssh_ok': bool,         # could we SSH at all?
          'vllm_ok': bool,        # is vLLM responding on the box's :8000?
          'model':  str | None,   # served-model-name if vllm_ok
          'error':  str | None,   # short human-readable diagnosis on failure
        }
    """
    import re
    host = (settings.get("host") or "").strip()
    user = (settings.get("user") or "ubuntu").strip() or "ubuntu"
    port = int(settings.get("port") or 22)
    key = (settings.get("key") or "").strip()
    if not host:
        return {"ssh_ok": False, "vllm_ok": False, "model": None,
                "error": "no host entered"}

    cmd = ["ssh"]
    if key:
        cmd += ["-i", key]
    cmd += [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/tmp/_box_probe_hosts",
        "-o", f"ConnectTimeout={int(timeout)}",
        "-p", str(port),
        f"{user}@{host}",
        # Print the curl response if vLLM is up; ALWAYS print SSH_OK as the
        # last line so we can tell SSH-success-but-no-vLLM apart from
        # SSH-failure (no output at all).
        "curl -sf -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 2048 || true; echo; echo __SSH_OK__",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        return {"ssh_ok": False, "vllm_ok": False, "model": None,
                "error": f"SSH connection timed out after ~{int(timeout)}s "
                         "(host unreachable, wrong port, firewall, etc.)"}
    except FileNotFoundError:
        return {"ssh_ok": False, "vllm_ok": False, "model": None,
                "error": "`ssh` command not found in PATH on the Mac"}

    out, err = r.stdout or "", r.stderr or ""
    ssh_ok = "__SSH_OK__" in out
    model = None
    m = re.search(r'"id"\s*:\s*"([^"]+)"', out)
    if m:
        model = m.group(1)
    vllm_ok = model is not None

    if not ssh_ok:
        # Distill the SSH error into something human-friendly.
        e = (err or out).strip()
        # Strip the harmless vast.ai banner so it doesn't clutter the message
        for noisy in (
            "Welcome to vast.ai. If authentication fails, try again",
            "Have fun!",
        ):
            e = e.replace(noisy, "").strip()
        # Trim very long traceback-y errors
        e = "\n".join(e.splitlines()[-4:])[:500] or f"exit code {r.returncode}"
        return {"ssh_ok": False, "vllm_ok": False, "model": None,
                "error": f"SSH failed: {e}"}

    if not vllm_ok:
        return {"ssh_ok": True, "vllm_ok": False, "model": None,
                "error": ("SSH reached the box, but nothing is listening on "
                          "127.0.0.1:8000 — vLLM isn't running there yet "
                          "(or it's serving on a different port).")}
    return {"ssh_ok": True, "vllm_ok": True, "model": model, "error": None}


def status_summary() -> dict:
    s = load_settings()
    pids = _running_keeper_pids()
    ep = is_endpoint_alive(local_port=int(s.get("local_port") or 8000))
    return {
        "settings": s,
        "keeper_running": bool(pids),
        "keeper_pids": pids,
        "endpoint": ep,
    }
