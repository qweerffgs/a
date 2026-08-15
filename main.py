import argparse
import atexit
import contextlib
import msvcrt
import ipaddress
import socket
import tempfile
import uuid
import hashlib
import html
import json
import logging
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "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")
NIM_BASE_URL   = "http://localhost:5000/v1"
MODEL          = "magni-preview"
MODEL_FALLBACKS = [                             
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
]

REQUESTS_PER_MINUTE = 19          
MAX_ITERATIONS      = 5000                                                            
MAX_RUNTIME_DAYS    = 0                                     
MAX_API_CALLS       = 0                                     
SHELL_TIMEOUT       = 120                                            
SHELL_TIMEOUT_MAX   = 1800                                                                    
MAX_OUTPUT_CHARS    = 6000                                                      
MAX_CONTEXT_CHARS   = 240_000                                                 
KEEP_RECENT_TURNS   = 14                                                       
REFLECT_EVERY       = 25                                                            
CHECKPOINT_EVERY    = 5                                                        
STALL_THRESHOLD     = 40                                                                  
TEMPERATURE         = 0.2
MAX_TOKENS          = 4096
VERIFIER_MAX_TURNS  = 10                                                       
GIT_CHECKPOINTS     = True                                       
WEBHOOK_URL         = ""                                                                        
WEBHOOK_EVERY_S     = 3600                                                

                                                                             
                                                                            
                                                                             
NVIDIA_API_KEYS     = [k.strip() for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]                                           

                                                                             
                                                
EMBED_ENABLED       = True
EMBED_MODEL         = "nvidia/nv-embedqa-e5-v5"                                        
EMBED_FALLBACKS     = ["baai/bge-m3", "nvidia/nv-embed-"]
MEMORY_TOP_K        = 6                                                               
MEMORY_MAX_ITEMS    = 6000                                                     
MEMORY_MIN_CHARS    = 40                                                

                                                                            
                                                                            
POWER_MODEL         = ""                                               

                                                                        
DELEGATE_MAX_TURNS  = 40                                                 
DELEGATE_MAX_DEPTH  = 2                                         

                                                                                  
MAINTENANCE_EVERY   = 20                                                  
DISK_MIN_FREE_MB    = 512                                                         
MEM_MIN_FREE_MB     = 192                                      
LOG_MAX_MB          = 40                                                        
JOBLOG_KEEP         = 40                                                   
BAK_KEEP            = 60                                             
GIT_GC_EVERY        = 400                                             

                                                                             
                                                 
WATCHDOG_TIMEOUT_S  = 3600                                                      

WORKDIR  = Path(os.environ.get("AGENT_WORKDIR", str(Path.cwd()))).expanduser().resolve()
LOG_FILE = WORKDIR / "agent.log"
NOTES    = WORKDIR / "NOTES.md"                                                        
STATE    = WORKDIR / "STATE.json"                                                         
CONVO    = WORKDIR / "CONVERSATION.json"                                  
STATUS   = WORKDIR / "STATUS.json"                                                             
JOBS_DIR = WORKDIR / "jobs"                                   
MEMORY   = WORKDIR / "MEMORY.jsonl"                                                    
EVENTS   = WORKDIR / "EVENTS.jsonl"                                                  
HEARTBEAT = WORKDIR / ".heartbeat"
LOCK_FILE = WORKDIR / ".agent.lock"
JOURNAL = WORKDIR / "ACTIONS.jsonl"
RUN_META = WORKDIR / "RUN.json"
MAX_EVENT_BYTES = 16_000_000
MAX_MEMORY_BYTES = 96_000_000
MAX_JOB_LOG_BYTES = 32_000_000
MAX_JOBS = 200
LOCK_HANDLE = None
RESTART_REQUESTED = threading.Event()                                 
                                                                               

START_T = time.time()
SHUTDOWN = False
AGENT_STATE: dict = {}                                                             
LIVE_PROCS: dict = {}                                                                   
CURRENT_MODEL = MODEL
CONV_REF: dict = {"messages": [], "summary": ""}                                          
STATS = {"api_calls": 0, "est_tokens": 0}
DELEGATE_DEPTH = 0                                                          

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("agent")


def _atomic_write(path: Path, data: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)
        raise


def _append_durable(path: Path, data: str, limit: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if limit and path.stat().st_size > limit:
        text = path.read_text(encoding="utf-8", errors="replace")
        _atomic_write(path, text[-limit // 2:])


def _acquire_lock() -> None:
    global LOCK_HANDLE
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_HANDLE = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        msvcrt.locking(LOCK_HANDLE.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise RuntimeError(f"another agent process holds {LOCK_FILE}")
    LOCK_HANDLE.seek(0)
    LOCK_HANDLE.truncate()
    LOCK_HANDLE.write(json.dumps({"pid": os.getpid(), "started": iso(), "workdir": str(WORKDIR)}))
    LOCK_HANDLE.flush()
    os.fsync(LOCK_HANDLE.fileno())


def _release_lock() -> None:
    global LOCK_HANDLE
    if LOCK_HANDLE:
        with contextlib.suppress(Exception):
            LOCK_HANDLE.seek(0)
            msvcrt.locking(LOCK_HANDLE.fileno(), msvcrt.LK_UNLCK, 1)
            LOCK_HANDLE.close()
        LOCK_HANDLE = None


def _state_schema(raw) -> dict:
    s = raw if isinstance(raw, dict) else {}
    defaults = {"goal": None, "expectation": None, "plan": [], "failures": [], "jobs": {}, "iteration": 0, "segments": 0, "done": False, "stats": {"api_calls": 0, "est_tokens": 0, "started": None}, "meta": {}, "exit_fails": {}}
    for key, value in defaults.items():
        if not isinstance(s.get(key), type(value)) and s.get(key) is not None:
            s[key] = value
        else:
            s.setdefault(key, value)
    if not isinstance(s["stats"], dict): s["stats"] = defaults["stats"].copy()
    for key, value in defaults["stats"].items(): s["stats"].setdefault(key, value)
    for key in ("iteration", "segments"):
        if not isinstance(s[key], int) or s[key] < 0: s[key] = 0
    return s


def _proc_start(pid: int):
    try:
        import ctypes
        import ctypes.wintypes
        h = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
        if not h:
            return None
        try:
            c = ctypes.wintypes.FILETIME()
            e = ctypes.wintypes.FILETIME()
            k = ctypes.wintypes.FILETIME()
            u = ctypes.wintypes.FILETIME()
            if ctypes.windll.kernel32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
                return str(c.dwLowDateTime + (c.dwHighDateTime << 32))
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:
        return None


def _owned_process(job: dict) -> bool:
    pid = job.get("pid")
    return isinstance(pid, int) and _pid_alive(pid) and str(job.get("proc_start")) == str(_proc_start(pid))


def _bounded_command(command: str, timeout: int, log_path: Path | None = None):
    timeout = max(1, timeout)
    capture = bytearray()
    limit = MAX_OUTPUT_CHARS * 4
    sink = None
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sink = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(command, shell=True, cwd=WORKDIR, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        deadline = time.monotonic() + timeout
        timed_out = False
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                with contextlib.suppress(OSError): proc.terminate()
                time.sleep(1)
                if proc.poll() is None:
                    with contextlib.suppress(OSError): proc.kill()
                break
            try:
                chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, 'read1') else proc.stdout.read(65536)
            except Exception:
                chunk = b''
            if chunk:
                if sink: sink.write(chunk)
                capture.extend(chunk)
                if len(capture) > limit: del capture[:-limit]
            else:
                time.sleep(0.05)
        if proc.stdout:
            with contextlib.suppress(Exception):
                while True:
                    chunk = proc.stdout.read(65536)
                    if not chunk: break
                    if sink: sink.write(chunk)
                    capture.extend(chunk)
                    if len(capture) > limit: del capture[:-limit]
        code = proc.wait(timeout=5) if proc.poll() is None else proc.returncode
        text = capture.decode("utf-8", errors="replace").strip()
        return (124 if timed_out else code), text + (f"\n(timed out after {timeout}s; process terminated)" if timed_out else "")
    finally:
        if sink: sink.close()


def _public_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("only plain https URLs are allowed")
    port = parsed.port or 443
    for item in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM):
        addr = ipaddress.ip_address(item[4][0])
        if not addr.is_global:
            raise ValueError("private, loopback, reserved, and link-local targets are blocked")
    return url


def _journal(phase: str, action: str, args: dict, result: str = "") -> None:
    redacted = dict(args)
    for key in ("token", "key", "password", "secret", "authorization"):
        if key in redacted: redacted[key] = "[redacted]"
    rec = {"t": iso(), "phase": phase, "action": action, "args": redacted, "result": result[:1000]}
    with contextlib.suppress(Exception): _append_durable(JOURNAL, json.dumps(rec, ensure_ascii=False) + "\n", 24_000_000)

def iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


                                                                               
                                                               
                                                                               
class RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self.last_call = 0.0
        self.backoff = 0.0

    def wait(self) -> None:
        gap = time.time() - self.last_call
        need = self.min_interval + self.backoff
        if gap < need:
            time.sleep(need - gap)
        self.last_call = time.time()

    def hit_429(self, retry_after=None) -> None:
        if retry_after:
            try:
                self.backoff = min(max(float(retry_after), 1.0), 180.0)
            except (TypeError, ValueError):
                self.backoff = min(self.backoff * 2 if self.backoff else 5.0, 180.0)
        else:
            self.backoff = min(self.backoff * 2 if self.backoff else 5.0, 180.0)
        log.warning("Rate limited by API — backing off %.0fs", self.backoff)

    def success(self) -> None:
        self.backoff = 0.0


rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)


                                                                               
                                       
                                                                            
                                                                                 
                                                                               
class KeyPool:
    def __init__(self, keys):
        self.keys = [k for k in keys if k]
        self.idx = 0
        self.cooldown = {}                                            

    def current(self) -> str:
        if not self.keys:
            return NVIDIA_API_KEY
        return self.keys[self.idx % len(self.keys)]

    def get(self) -> str:
        """Current key, skipping cooled-down ones; waits if all are cooling."""
        if not self.keys:
            return NVIDIA_API_KEY
        now = time.time()
        for _ in range(len(self.keys)):
            k = self.keys[self.idx % len(self.keys)]
            if self.cooldown.get(k, 0) <= now:
                return k
            self.idx += 1
        soonest = min(self.cooldown.get(k, 0) for k in self.keys)
        wait = max(soonest - now, 5.0)
        log.warning("All %d API keys cooling down — waiting %.0fs", len(self.keys), wait)
        time.sleep(min(wait, 600))
        self.idx += 1
        return self.keys[self.idx % len(self.keys)]

    def report(self, key: str, kind: str) -> None:
        """kind: 'ok' | 'rate' | 'dead'  — circuit breaker per key."""
        if not self.keys or key not in self.keys:
            return
        if kind == "ok":
            self.cooldown.pop(key, None)
        elif kind == "rate":
            self.cooldown[key] = time.time() + 120
            self.idx += 1
            log.warning("Key ...%s rate-limited — rotating (cooldown 120s)", key[-6:])
        elif kind == "dead":
            self.cooldown[key] = time.time() + 3600
            self.idx += 1
            log.error("Key ...%s rejected (auth/quota) — sidelined for 1h", key[-6:])


key_pool = KeyPool(NVIDIA_API_KEYS)


                                                                               
                                                                   
                                                                               
class ModelError(Exception):
    """The model itself is rejected (400/404) — trigger a fallback."""


def nim_chat(messages, model=None, retries=8, max_tokens=None, temperature=None) -> str:
    url = f"{NIM_BASE_URL}/chat/completions"
    payload = {
        "model": model or CURRENT_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or MAX_TOKENS,
        "stream": False,
    }
    body = json.dumps(payload).encode()

    for attempt in range(retries):
        rate_limiter.wait()
        api_key = key_pool.get()
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            rate_limiter.success()
            key_pool.report(api_key, "ok")
            content = data["choices"][0]["message"]["content"] or ""
            STATS["api_calls"] += 1
            STATS["est_tokens"] += len(body) // 4 + len(content) // 4
            return content
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode(errors="replace")[:500]
            except Exception:
                pass
            if exc.code == 429:
                rate_limiter.hit_429(exc.headers.get("Retry-After") if exc.headers else None)
                key_pool.report(api_key, "rate")
                continue
            log.error("NIM HTTP %s: %s", exc.code, detail)
            if exc.code in (401, 403):
                key_pool.report(api_key, "dead")
                continue
            if exc.code in (400, 404) and "model" in detail.lower():
                raise ModelError(f"{exc.code}: {detail}")
            if exc.code in (500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(2 ** attempt, 60))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            log.error("NIM call failed (%s); retry %d/%d", exc, attempt + 1, retries)
            time.sleep(min(2 ** attempt, 60))

    raise RuntimeError("NIM API unreachable after retries")


def llm_chat(messages, **kwargs) -> str:
    """Chat with automatic model fallback on model-level rejection."""
    global CURRENT_MODEL
    try:
        return nim_chat(messages, model=CURRENT_MODEL, **kwargs)
    except ModelError as exc:
        last = exc
        for fb in MODEL_FALLBACKS:
            if fb == CURRENT_MODEL:
                continue
            try:
                log.warning("Model fallback: %s", fb)
                CURRENT_MODEL = fb
                return nim_chat(messages, model=fb, **kwargs)
            except ModelError as next_exc:
                last = next_exc
        raise last


def llm_chat_power(messages, **kwargs) -> str:
    """: escalate to POWER_MODEL for high-stakes calls (verification,
    compaction, stuck turns). Falls back to the normal chain if unset/failing."""
    if not POWER_MODEL:
        return llm_chat(messages, **kwargs)
    try:
        return nim_chat(messages, model=POWER_MODEL, **kwargs)
    except Exception as exc:
        log.warning("POWER_MODEL %s failed (%s) — using standard chain", POWER_MODEL, exc)
        return llm_chat(messages, **kwargs)


                                                                               
                                                                            
                                                                               
EMBED_STATE = {"available": EMBED_ENABLED, "failures": 0, "model": EMBED_MODEL}


def nim_embed(texts):
    """Embed a list of strings. Returns list[list[float]] or None on failure.
    Tries EMBED_MODEL then EMBED_FALLBACKS; disables itself after repeated
    failures so the agent never stalls on a dead embedding endpoint."""
    if not EMBED_STATE["available"]:
        return None
    models = [EMBED_STATE["model"]] + [m for m in EMBED_FALLBACKS if m != EMBED_STATE["model"]]
    for model in models:
        payload = {
            "model": model,
            "input": [t[:2000] for t in texts],                        
            "input_type": "passage",
            "encoding_format": "float",
            "truncate": "END",
        }
        body = json.dumps(payload).encode()
        for attempt in range(3):
            rate_limiter.wait()
            req = urllib.request.Request(
                f"{NIM_BASE_URL}/embeddings", data=body,
                headers={"Authorization": f"Bearer {key_pool.get()}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode())
                vecs = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
                STATS["api_calls"] += 1
                EMBED_STATE["model"] = model
                EMBED_STATE["failures"] = 0
                return vecs
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    rate_limiter.hit_429(exc.headers.get("Retry-After") if exc.headers else None)
                    continue
                log.warning("Embedding model %s HTTP %s — trying next", model, exc.code)
                break                                    
            except Exception as exc:
                log.warning("Embedding call failed (%s); retry %d/3", exc, attempt + 1)
                time.sleep(min(2 ** attempt, 20))
    EMBED_STATE["failures"] += 1
    if EMBED_STATE["failures"] >= 3:
        EMBED_STATE["available"] = False
        log.error("Embeddings disabled after repeated failures — keyword recall fallback active")
    return None


                                                                               
                                                                              
                                                                             
                                                               
                                                                               
_MEM_CACHE = {"loaded": False, "items": []}


def _mem_load():
    if _MEM_CACHE["loaded"]:
        return _MEM_CACHE["items"]
    items = []
    if MEMORY.is_file():
        try:
            for line in MEMORY.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and rec.get("text"):
                        items.append(rec)
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    _MEM_CACHE["items"] = items
    _MEM_CACHE["loaded"] = True
    return items


def _mem_append(rec: dict) -> None:
    _mem_load().append(rec)
    try:
        with open(MEMORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("memory append failed: %s", exc)


def _mem_prune() -> None:
    items = _mem_load()
    if len(items) > MEMORY_MAX_ITEMS:
        del items[: len(items) - MEMORY_MAX_ITEMS]
        try:
            MEMORY.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items),
                              encoding="utf-8")
            log.info("[memory] pruned to %d items", len(items))
        except OSError:
            pass


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _kw_score(query: str, text: str) -> float:
    """BM25-lite: term-overlap weighted by rarity-ish length penalty."""
    q = set(re.findall(r"[a-z0-9_]+", query.lower()))
    t = re.findall(r"[a-z0-9_]+", text.lower())
    if not q or not t:
        return 0.0
    tf = {}
    for w in t:
        tf[w] = tf.get(w, 0) + 1
    score = sum(1.0 + math.log(tf[w]) for w in q if w in tf)
    return score / (1.0 + len(t) / 120.0)


def memory_store(text: str, kind: str = "fact") -> str:
    """Embed and persist a durable memory. Dedupes on near-identical text."""
    text = text.strip()
    if len(text) < MEMORY_MIN_CHARS:
        return f"ERROR: memory too short (<{MEMORY_MIN_CHARS} chars) — make it substantive"
    items = _mem_load()
    for rec in items[-200:]:
        if rec.get("text", "") == text:
            return "OK: already stored (duplicate skipped)"
    vec = None
    if EMBED_STATE["available"]:
        vecs = nim_embed([text])
        if vecs:
            vec = vecs[0]
    _mem_append({"t": iso(), "kind": kind, "text": text, "vec": vec})
    _mem_prune()
    log.info("[memory] stored (%s, %s): %.80s", kind, "vector" if vec else "keyword", text)
    return f"OK: remembered ({'semantic vector' if vec else 'keyword-only'})"


def memory_recall(query: str, top_k: int = None):
    """Top-k memories most relevant to query. Vector cosine if embeddings
    exist for both sides, else keyword scoring. Never raises."""
    items = _mem_load()
    if not items:
        return []
    k = top_k or MEMORY_TOP_K
    qvec = None
    if EMBED_STATE["available"]:
        vecs = nim_embed([query])
        if vecs:
            qvec = vecs[0]
    scored = []
    for rec in items:
        if qvec and rec.get("vec"):
            s = _cosine(qvec, rec["vec"])
        else:
            s = _kw_score(query, rec["text"])
        if s > 0:
            scored.append((s, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def memory_context_block(query: str) -> str:
    """Rendered for the system prompt each turn."""
    try:
        recs = memory_recall(query)
    except Exception as exc:
        log.warning("memory recall failed: %s", exc)
        return ""
    if not recs:
        return ""
    lines = ["RECALLED LONG-TERM MEMORIES (most relevant to current work):"]
    for r in recs:
        lines.append(f"  - [{r.get('kind', 'fact')} @ {r.get('t', '?')[:10]}] {r['text'][:400]}")
    return "\n".join(lines)


                                                                               
                                                                          
                                                                               
def log_event(kind: str, text: str, remember: bool = False) -> None:
    rec = {"t": iso(), "kind": kind, "text": text[:800]}
    try:
        with open(EVENTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if remember:
        memory_store(f"[{kind}] {text}", kind="event")


                                                                               
                                         
                                                                             
                                                                          
                                                                               
def _disk_free_mb() -> float:
    try:
        import shutil
        return shutil.disk_usage(WORKDIR).free / 1e6
    except OSError:
        return 9999.0


def _mem_free_mb() -> float:
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / 1e6
    except Exception:
        return 9999.0


def resource_warnings() -> str:
    """Advisory text injected into the system prompt when resources run low."""
    warns = []
    d = _disk_free_mb()
    if d < DISK_MIN_FREE_MB:
        warns.append(f"LOW DISK: only {d:.0f}MB free — clean up artifacts NOW (rm old logs/dumps)")
    m = _mem_free_mb()
    if m < MEM_MIN_FREE_MB:
        warns.append(f"LOW MEMORY: only {m:.0f}MB available — avoid heavy commands; prefer background jobs")
    return "\n".join(warns)


def run_maintenance(state: dict) -> None:
    """Housekeeping pass: rotate logs, prune .bak/job logs, gc git, bound memory."""
    freed = []

                                        
    try:
        if LOG_FILE.is_file() and LOG_FILE.stat().st_size > LOG_MAX_MB * 1e6:
            data = LOG_FILE.read_text(encoding="utf-8", errors="replace")
            keep = int(LOG_MAX_MB * 1e6 * 0.25)
            LOG_FILE.write_text(f"--- rotated {iso()} ---\n" + data[-keep:], encoding="utf-8")
            freed.append("agent.log rotated")
    except OSError as exc:
        log.warning("maintenance: log rotate failed: %s", exc)

                                                 
    try:
        baks = sorted(WORKDIR.rglob("*.bak"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in baks[BAK_KEEP:]:
            p.unlink()
        if len(baks) > BAK_KEEP:
            freed.append(f"{len(baks) - BAK_KEEP} old .bak files")
    except OSError:
        pass

                                                           
    try:
        if JOBS_DIR.is_dir():
            logs = sorted(JOBS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in logs[JOBLOG_KEEP:]:
                p.unlink()
            if len(logs) > JOBLOG_KEEP:
                freed.append(f"{len(logs) - JOBLOG_KEEP} old job logs")
    except OSError:
        pass

                         
    if GIT_CHECKPOINTS and (WORKDIR / ".git").is_dir():
        last_gc = state.setdefault("meta", {}).get("last_git_gc_iter", 0)
        if state.get("iteration", 0) - last_gc >= GIT_GC_EVERY:
            code, out = _sh("git gc --prune=now", timeout=300)
            state["meta"]["last_git_gc_iter"] = state.get("iteration", 0)
            freed.append(f"git gc (exit {code})")

    _mem_prune()

    if freed:
        log.info("[maintenance] %s", "; ".join(freed))
        log_event("maintenance", "; ".join(freed))
    warn = resource_warnings()
    if warn:
        log.warning("[maintenance] %s", warn.replace("\n", " | "))
        if _disk_free_mb() < DISK_MIN_FREE_MB:
            webhook_ping(f"⚠ Agent disk low: {_disk_free_mb():.0f}MB free on {WORKDIR}")


                                                                               
                                                                      
                                                         
                                                                               
_HEARTBEAT = {"t": time.time()}


def heartbeat() -> None:
    _HEARTBEAT["t"] = time.time()
    try:
        HEARTBEAT.write_text(iso(), encoding="utf-8")
    except OSError:
        pass


def _watchdog_loop() -> None:
    while not SHUTDOWN:
        time.sleep(60)
        if WATCHDOG_TIMEOUT_S <= 0:
            continue
        silent = time.time() - _HEARTBEAT["t"]
        if silent > WATCHDOG_TIMEOUT_S:
            log.error("WATCHDOG: loop frozen for %.0fs — saving state and re-executing", silent)
            log_event("watchdog", f"frozen {silent:.0f}s — self-heal restart", remember=True)
            try:
                if AGENT_STATE:
                    save_state(AGENT_STATE)
                if CONV_REF["messages"]:
                    save_conversation(CONV_REF["messages"], CONV_REF["summary"])
                for h in logging.getLogger().handlers + log.handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass
            except Exception:
                pass
            os._exit(75)


def start_watchdog() -> None:
    if WATCHDOG_TIMEOUT_S > 0:
        t = threading.Thread(target=_watchdog_loop, daemon=True, name="watchdog")
        t.start()
        log.info("Watchdog armed (freeze timeout %ds)", WATCHDOG_TIMEOUT_S)


                                                                               
                                                                         
                                                                               
def repair_json(text: str) -> str:
    """Best-effort fixes for common LLM JSON sins: trailing commas, raw
    newlines/tabs inside strings, single quotes, missing closing braces."""
    t = text.strip()
                                                         
    out, in_str, esc = [], False, False
    for c in t:
        if in_str and not esc and c in "\n\t":
            out.append("\\n" if c == "\n" else "\\t")
            continue
        out.append(c)
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
    t = "".join(out)
    t = re.sub(r",(\s*[}\]])", r"\1", t)                           
                                                     
    depth, in_str, esc = [], False, False
    for c in t:
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c in "{[":
                depth.append(c)
            elif c in "}]":
                if depth and ((c == "}") == (depth[-1] == "{")):
                    depth.pop()
    if in_str:
        t += '"'
    t += "".join("}" if c == "{" else "]" for c in reversed(depth))
    return t


                                                                               
          
                                                                               
def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        return text[:half] + f"\n...[truncated {len(text) - MAX_OUTPUT_CHARS} chars]...\n" + text[-half:]
    return text


def _safe_path(path: str):
    p = (WORKDIR / path).resolve()
    if not str(p).startswith(str(WORKDIR.resolve())):
        return None
    return p


def _sh(command: str, timeout: int = 60):
    try:
        proc = subprocess.run(command, shell=True, cwd=WORKDIR, timeout=timeout,
                              capture_output=True, text=True, errors="replace")
        return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {timeout}s)"


def _pid_alive(pid: int) -> bool:
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:
        return False


                                                                               
                            
                                                                               
def tool_run_shell(args: dict) -> str:
    command = args["command"]
    try:
        timeout = int(args.get("timeout", SHELL_TIMEOUT))
    except (TypeError, ValueError):
        timeout = SHELL_TIMEOUT
    timeout = min(max(timeout, 1), SHELL_TIMEOUT_MAX)
    log.info("[shell] (%ds) %s", timeout, command)
    try:
        proc = subprocess.run(command, shell=True, cwd=WORKDIR, timeout=timeout,
                              capture_output=True, text=True, errors="replace")
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return _truncate(f"exit={proc.returncode}\n{out or '(no output)'}")
    except subprocess.TimeoutExpired:
        return (f"ERROR: command timed out after {timeout}s. "
                f"If it is a long-running task, rerun it with run_background and poll with check_jobs.")


def tool_run_background(args: dict) -> str:
    command = args["command"]
    JOBS_DIR.mkdir(exist_ok=True)
    job_id = f"job-{int(time.time())}-{len(AGENT_STATE.get('jobs', {})) + 1}"
    logpath = JOBS_DIR / f"{job_id}.log"
    logf = open(logpath, "ab")
    proc = subprocess.Popen(command, shell=True, cwd=WORKDIR,
                            stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    LIVE_PROCS[job_id] = proc
    AGENT_STATE.setdefault("jobs", {})[job_id] = {
        "pid": proc.pid, "command": command[:500],
        "log": str(logpath.relative_to(WORKDIR)), "started": iso(), "status": "running",
    }
    save_state(AGENT_STATE)
    log.info("[bg] %s (pid %d): %s", job_id, proc.pid, command)
    return (f"OK: started {job_id} (pid {proc.pid}). Output streams to {logpath.name}. "
            f"Poll with check_jobs; inspect with read_file on the log.")


def tool_check_jobs(args: dict) -> str:
    jobs = AGENT_STATE.setdefault("jobs", {})
    if not jobs:
        return "(no background jobs started)"
    lines = []
    changed = False
    for jid, j in jobs.items():
        status = j.get("status", "unknown")
        if status == "running":
            if jid in LIVE_PROCS:
                rc = LIVE_PROCS[jid].poll()
                if rc is not None:
                    status = f"exited (code {rc})"
                    j["status"] = status
                    changed = True
            elif not _pid_alive(j.get("pid", -1)):
                status = "exited (process gone)"
                j["status"] = status
                changed = True
        tail = ""
        logrel = j.get("log")
        if logrel and (WORKDIR / logrel).is_file():
            try:
                data = (WORKDIR / logrel).read_text(encoding="utf-8", errors="replace")
                tail = "\n    log tail: " + data[-400:].replace("\n", "\n    ") if data else ""
            except Exception:
                pass
        lines.append(f"- {jid}: {status} | pid {j.get('pid')} | started {j.get('started')}\n"
                     f"    cmd: {j.get('command', '')[:200]}{tail}")
    if changed:
        save_state(AGENT_STATE)
    return _truncate("\n".join(lines))


def tool_kill_job(args: dict) -> str:
    job_id = args["job_id"]
    j = AGENT_STATE.setdefault("jobs", {}).get(job_id)
    if not j:
        return f"ERROR: unknown job '{job_id}'"
    pid = j.get("pid", -1)
    if not _pid_alive(pid):
        j["status"] = "exited"
        save_state(AGENT_STATE)
        return f"OK: {job_id} was not running."
    try:
        proc = LIVE_PROCS.get(job_id)
        if proc:
            proc.terminate()
            time.sleep(2)
            if _pid_alive(pid):
                proc.kill()
        else:
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x0001, False, pid)
            if h:
                ctypes.windll.kernel32.TerminateProcess(h, 1)
                ctypes.windll.kernel32.CloseHandle(h)
        j["status"] = "killed"
        save_state(AGENT_STATE)
        log.info("[bg] killed %s (pid %d)", job_id, pid)
        return f"OK: killed {job_id} (pid {pid})"
    except OSError as exc:
        return f"ERROR: {exc}"


def tool_read_file(args: dict) -> str:
    p = _safe_path(args["path"])
    if p is None:
        return "ERROR: path escapes WORKDIR"
    if not p.is_file():
        return f"ERROR: no such file: {args['path']}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: {exc}"
    start = args.get("start_line")
    end = args.get("end_line")
    if start or end:
        lines = text.splitlines()
        s = max(int(start or 1), 1)
        e = int(end) if end else len(lines)
        chunk = "\n".join(lines[s - 1:e])
        return _truncate(f"(lines {s}-{min(e, len(lines))} of {len(lines)})\n{chunk}")
    return _truncate(text)


def tool_write_file(args: dict) -> str:
    p = _safe_path(args["path"])
    if p is None:
        return "ERROR: path escapes WORKDIR"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"], encoding="utf-8")
    log.info("[write] %s (%d chars)", args["path"], len(args["content"]))
    return f"OK: wrote {len(args['content'])} chars to {args['path']}"


def tool_append_file(args: dict) -> str:
    p = _safe_path(args["path"])
    if p is None:
        return "ERROR: path escapes WORKDIR"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(args["content"])
    log.info("[append] %s (%d chars)", args["path"], len(args["content"]))
    return f"OK: appended {len(args['content'])} chars to {args['path']}"


def tool_edit_file(args: dict) -> str:
    p = _safe_path(args["path"])
    if p is None:
        return "ERROR: path escapes WORKDIR"
    if not p.is_file():
        return f"ERROR: no such file: {args['path']}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if args["old"] not in text:
        return "ERROR: 'old' string not found in file"
                                                  
    try:
        (p.parent / (p.name + ".bak")).write_text(text, encoding="utf-8")
    except Exception:
        pass
    p.write_text(text.replace(args["old"], args["new"], 1), encoding="utf-8")
    log.info("[edit] %s", args["path"])
    return "OK: edit applied (backup at %s.bak)" % args["path"]


def tool_list_dir(args: dict) -> str:
    p = _safe_path(args.get("path", "."))
    if p is None or not p.is_dir():
        return f"ERROR: not a directory inside WORKDIR: {args.get('path')}"
    try:
        depth = int(args.get("depth", 3))
    except (TypeError, ValueError):
        depth = 3
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", "jobs"}
    entries = []
    for root, dirs, files in os.walk(p):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        rel = os.path.relpath(root, p)
        level = 0 if rel == "." else rel.count(os.sep) + 1
        if level >= depth:
            dirs[:] = []
        prefix = "" if rel == "." else rel + "/"
        for d in dirs:
            entries.append(prefix + d + "/")
        for f in sorted(files):
            entries.append(prefix + f)
        if len(entries) >= 400:
            entries.append("...(truncated)")
            break
    return "\n".join(entries[:400]) or "(empty)"


def tool_search_files(args: dict) -> str:
    base = _safe_path(args.get("path", "."))
    if base is None or not base.is_dir():
        return f"ERROR: not a directory inside WORKDIR: {args.get('path')}"
    flags = re.I if args.get("ignore_case", True) else 0
    try:
        rx = re.compile(args["pattern"], flags)
    except re.error as exc:
        return f"ERROR: bad regex: {exc}"
    max_results = min(int(args.get("max_results", 40)), 200)
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", "jobs"}
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            fp = Path(root) / name
            try:
                if fp.stat().st_size > 2_000_000:
                    continue
                for n, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        rel = fp.relative_to(WORKDIR)
                        hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                        if len(hits) >= max_results:
                            return "\n".join(hits) + "\n...(truncated)"
            except OSError:
                continue
    return "\n".join(hits) or "(no matches)"


def tool_fetch_url(args: dict) -> str:
    url = args["url"]
    log.info("[fetch] %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 agent"})
        with urllib.request.urlopen(req, timeout=40) as resp:
            raw = resp.read(3_000_000).decode("utf-8", errors="replace")
        text = re.sub(r"<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>", " ", raw, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return _truncate(text.strip())
    except Exception as exc:
        return f"ERROR: {exc}"


def tool_web_search(args: dict) -> str:
    query = args["query"]
    log.info("[search] %s", query)
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(1_000_000).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: {exc}"
    results = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if "uddg" in qs:
            href = qs["uddg"][0]
        results.append(f"- {html.unescape(title).strip()}\n  {href}")
        if len(results) >= 8:
            break
    if not results:
        return "(no results parsed — try fetch_url on a known URL instead)"
    return "\n".join(results) + "\n\nUse fetch_url on a result to read the page."


def tool_update_notes(args: dict) -> str:
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(f"\n\n### {iso()}\n{args['markdown']}")
                                                                        
    try:
        if NOTES.stat().st_size > 200_000:
            data = NOTES.read_text(encoding="utf-8", errors="replace")
            NOTES.write_text("# NOTES (rotated — older entries discarded)\n" + data[-100_000:],
                             encoding="utf-8")
            log.info("[notes] rotated NOTES.md")
    except OSError:
        pass
    return "OK: notes saved"


def tool_update_plan(args: dict) -> str:
    steps = args.get("steps")
    if not isinstance(steps, list):
        return "ERROR: 'steps' must be a list of {step, status} objects"
    norm = []
    for s in steps[:50]:
        if isinstance(s, str):
            norm.append({"step": s[:300], "status": "pending"})
        elif isinstance(s, dict):
            status = str(s.get("status", "pending"))
            if status not in ("pending", "in_progress", "done"):
                status = "pending"
            norm.append({"step": str(s.get("step", ""))[:300], "status": status})
    AGENT_STATE["plan"] = norm
    save_state(AGENT_STATE)
    log.info("[plan] updated (%d steps)", len(norm))
    return f"OK: plan updated ({len(norm)} steps)"


def tool_git_checkpoint(args: dict) -> str:
    if not GIT_CHECKPOINTS:
        return "ERROR: git checkpoints disabled in config"
    msg = args.get("message") or f"agent checkpoint iter {AGENT_STATE.get('iteration', '?')}"
    if not (WORKDIR / ".git").is_dir():
        code, out = _sh("git init", timeout=30)
        if code != 0:
            return f"ERROR: git init failed: {out[:300]}"
        (WORKDIR / ".gitignore").write_text(
            "jobs/\n__pycache__/\n*.pyc\n*.bak\nagent.log\nSTATUS.json\nCONVERSATION.json\n",
            encoding="utf-8")
        _sh('git config user.email "agent@local" && git config user.name "autonomous-agent"')
    code, out = _sh("git add -A && git commit -m " + shlex.quote(msg), timeout=120)
    if code != 0 and ("nothing to commit" in out or "no changes added" in out):
        return "OK: nothing new to commit"
    log.info("[git] checkpoint: %s (exit %d)", msg, code)
    return _truncate(f"exit={code}\n{out or '(no output)'}")


def tool_remember(args: dict) -> str:
    """: store a durable fact/decision/gotcha in semantic long-term memory."""
    return memory_store(str(args.get("text", "")), kind=str(args.get("kind", "fact")))


def tool_recall(args: dict) -> str:
    """: search semantic long-term memory."""
    query = str(args.get("query", ""))
    if not query:
        return "ERROR: 'query' is required"
    recs = memory_recall(query, top_k=min(int(args.get("top_k", MEMORY_TOP_K)), 20))
    if not recs:
        return "(no relevant memories found)"
    return "\n".join(f"- [{r.get('kind', 'fact')} @ {r.get('t', '?')}] {r['text']}" for r in recs)


def tool_delegate(args: dict) -> str:
    """: hand a well-scoped subtask to an isolated sub-agent (fresh context)."""
    global DELEGATE_DEPTH
    if DELEGATE_DEPTH >= DELEGATE_MAX_DEPTH:
        return "ERROR: delegation depth limit reached — do this subtask yourself"
    task = str(args.get("task", "")).strip()
    if not task:
        return "ERROR: 'task' is required"
    context = str(args.get("context", ""))
    try:
        max_turns = min(int(args.get("max_turns", DELEGATE_MAX_TURNS)), DELEGATE_MAX_TURNS)
    except (TypeError, ValueError):
        max_turns = DELEGATE_MAX_TURNS
    log.info("[delegate] depth %d, task: %.120s", DELEGATE_DEPTH + 1, task)
    log_event("delegate", f"spawned worker: {task[:200]}")
    DELEGATE_DEPTH += 1
    try:
        result = run_subagent(task, context, max_turns)
    finally:
        DELEGATE_DEPTH -= 1
    log_event("delegate", f"worker finished: {task[:120]} -> {result[:200]}")
    return _truncate(f"SUB-AGENT RESULT:\n{result}")


def run_subagent(task: str, context: str, max_turns: int) -> str:
    """An isolated agent loop with fresh context. The worker gets the full tool
    set (minus goal_complete) and returns only a digest — the main loop's
    context stays clean no matter how much the worker explores."""
    sys_prompt = f"""You are a SUB-AGENT working inside a larger autonomous mission on a Windows 11 machine (cwd: {WORKDIR}).
Complete the assigned subtask, then report back. You have fresh context — everything
you need is in the subtask and context below. Work autonomously; verify with real
command output. Do NOT call goal_complete — only the parent agent may do that.

SUBTASK:
{task}

CONTEXT FROM PARENT AGENT:
{context or "(none)"}

{TOOL_SPEC}"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Begin the subtask now. When finished, reply with "
            '{"action": "report", "args": {"summary": "what you did, key outputs, file paths, anything the parent must know"}}'},
    ]
    for turn in range(max_turns):
        heartbeat()
        try:
            reply = llm_chat(messages)
        except Exception as exc:
            log.error("[subagent] model call failed: %s", exc)
            time.sleep(20)
            continue
        action = extract_json_object(reply)
        if action is None:
            action = extract_json_object(repair_json(reply))
        if action is None or "action" not in action:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                "Reply with EXACTLY one JSON action object — no prose, no fences."})
            continue
        act = str(action.get("action", ""))
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if act == "report":
            summary = str(args.get("summary", "")).strip()
            log.info("[subagent] reported: %.200s", summary)
            return summary or "(sub-agent returned an empty report)"
        if act == "goal_complete":
            result = "ERROR: sub-agents cannot call goal_complete. Use report when done."
        elif act not in TOOLS:
            result = f"ERROR: unknown action '{act}'"
        else:
            try:
                result = TOOLS[act](args)
            except KeyError as exc:
                result = f"ERROR: missing arg {exc} for action '{act}'"
            except Exception as exc:
                result = f"ERROR executing {act}: {exc}"
        log.info("[subagent] %s -> %.150s", act, result)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"TOOL RESULT ({act}):\n{result}"})
        if context_size(messages) > MAX_CONTEXT_CHARS:
            messages, _ = compact_context(messages, "")
    return f"(sub-agent exhausted its {max_turns}-turn budget without reporting; partial progress may exist in the workspace)"


def tool_restart_self(args: dict) -> str:
    """Save everything and re-exec — used after the agent improves its own code."""
    reason = args.get("reason", "")
    log.warning("[restart] agent re-executing itself: %s", reason)
    AGENT_STATE["restart_count"] = AGENT_STATE.get("restart_count", 0) + 1
    save_state(AGENT_STATE)
    if CONV_REF["messages"]:
        save_conversation(CONV_REF["messages"], CONV_REF["summary"])
    for h in logging.getLogger().handlers + log.handlers:
        try:
            h.flush()
        except Exception:
            pass
    RESTART_REQUESTED.set()
    return "ERROR: exec failed"               


def _safe_path(path: str):
    try:
        root = WORKDIR.resolve()
        p = (root / path).resolve(strict=False)
        if p != root and root not in p.parents:
            return None
        return p
    except (OSError, ValueError):
        return None


def _sh(command: str, timeout: int = 60):
    return _bounded_command(command, min(max(1, timeout), SHELL_TIMEOUT_MAX))


def tool_run_shell(args: dict) -> str:
    command = str(args["command"])
    try: timeout = min(max(1, int(args.get("timeout", SHELL_TIMEOUT))), SHELL_TIMEOUT_MAX)
    except (TypeError, ValueError): timeout = SHELL_TIMEOUT
    code, output = _bounded_command(command, timeout)
    return _truncate(f"exit={code}\n{output or '(no output)'}")


def tool_run_background(args: dict) -> str:
    command = str(args["command"])
    jobs = AGENT_STATE.setdefault("jobs", {})
    if sum(j.get("status") == "running" for j in jobs.values()) >= 32:
        return "ERROR: background job limit reached"
    JOBS_DIR.mkdir(exist_ok=True)
    job_id = f"job-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    logpath = JOBS_DIR / f"{job_id}.log"
    logf = open(logpath, "ab", buffering=0)
    try:
        proc = subprocess.Popen(command, shell=True, cwd=WORKDIR, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    finally:
        logf.close()
    LIVE_PROCS[job_id] = proc
    jobs[job_id] = {"pid": proc.pid, "proc_start": _proc_start(proc.pid), "command": command[:500], "command_hash": hashlib.sha256(command.encode()).hexdigest(), "log": str(logpath.relative_to(WORKDIR)), "started": iso(), "status": "running"}
    save_state(AGENT_STATE)
    return f"OK: started {job_id} (pid {proc.pid}); poll with check_jobs"


def tool_check_jobs(args: dict) -> str:
    jobs = AGENT_STATE.setdefault("jobs", {})
    lines, changed = [], False
    for jid, job in list(jobs.items()):
        if job.get("status") == "running":
            proc = LIVE_PROCS.get(jid)
            if proc and proc.poll() is not None:
                job["status"] = f"exited (code {proc.returncode})"; LIVE_PROCS.pop(jid, None); changed = True
            elif not _owned_process(job):
                job["status"] = "exited (process unavailable)"; LIVE_PROCS.pop(jid, None); changed = True
        tail = ""
        lp = WORKDIR / str(job.get("log", ""))
        if lp.is_file():
            try:
                with open(lp, "rb") as f:
                    f.seek(max(0, lp.stat().st_size - 800)); tail = f.read().decode("utf-8", errors="replace")
            except OSError: pass
        lines.append(f"- {jid}: {job.get('status')} | pid {job.get('pid')}\n  {job.get('command', '')[:200]}\n  {tail[:800]}")
    if changed: save_state(AGENT_STATE)
    return _truncate("\n".join(lines) or "(no background jobs started)")


def tool_kill_job(args: dict) -> str:
    job = AGENT_STATE.setdefault("jobs", {}).get(args["job_id"])
    if not job: return "ERROR: unknown job"
    if not _owned_process(job):
        job["status"] = "exited (process unavailable)"; save_state(AGENT_STATE); return "OK: job was not running"
    try:
        proc = LIVE_PROCS.get(args["job_id"])
        if proc:
            proc.terminate(); time.sleep(2)
            if _owned_process(job): proc.kill()
        else:
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x0001, False, job["pid"])
            if h:
                ctypes.windll.kernel32.TerminateProcess(h, 1)
                ctypes.windll.kernel32.CloseHandle(h)
        job["status"] = "killed"; LIVE_PROCS.pop(args["job_id"], None); save_state(AGENT_STATE)
        return "OK: job terminated"
    except OSError as exc: return f"ERROR: {exc}"


def tool_fetch_url(args: dict) -> str:
    url = _public_url(str(args["url"]))
    opener = urllib.request.build_opener(urllib.request.HTTPHandler(), urllib.request.HTTPSHandler(), urllib.request.HTTPErrorProcessor())
    for _ in range(4):
        req = urllib.request.Request(url, headers={"User-Agent": "autonomous-agent/1.0"})
        try:
            with opener.open(req, timeout=30) as resp:
                if resp.status in (301,302,303,307,308):
                    url = _public_url(urllib.parse.urljoin(url, resp.headers["Location"])); continue
                if int(resp.headers.get("Content-Length", "0") or 0) > 3_000_000: return "ERROR: response exceeds size limit"
                raw = resp.read(3_000_000 + 1)
                if len(raw) > 3_000_000: return "ERROR: response exceeds size limit"
                text = raw.decode("utf-8", errors="replace")
                text = re.sub(r"<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>", " ", text, flags=re.S|re.I)
                return _truncate(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip())
        except urllib.error.HTTPError as exc:
            if exc.code in (301,302,303,307,308) and exc.headers.get("Location"):
                url = _public_url(urllib.parse.urljoin(url, exc.headers["Location"])); continue
            return f"ERROR: HTTP {exc.code}"
        except Exception as exc: return f"ERROR: {exc}"
    return "ERROR: too many redirects"


def load_state() -> dict:
    try: raw = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    except Exception:
        with contextlib.suppress(Exception): STATE.replace(STATE.with_name(f"STATE.corrupt.{int(time.time())}.json"))
        raw = {}
    return _state_schema(raw)


def save_state(state: dict) -> None:
    state["stats"]["api_calls"] = STATS["api_calls"]
    state["stats"]["est_tokens"] = STATS["est_tokens"]
    _atomic_write(STATE, json.dumps(_state_schema(state), indent=2, ensure_ascii=False))


def save_conversation(messages, summary: str) -> None:
    compact = {"messages": messages, "summary": summary, "saved": iso()}
    _atomic_write(CONVO, json.dumps(compact, ensure_ascii=False))


def log_event(kind: str, text: str, remember: bool = False) -> None:
    with contextlib.suppress(Exception): _append_durable(EVENTS, json.dumps({"t": iso(), "kind": kind, "text": text[:800]}, ensure_ascii=False) + "\n", MAX_EVENT_BYTES)
    if remember: memory_store(f"[{kind}] {text}", kind="event")


def run_maintenance(state: dict) -> None:
    for target, limit in ((EVENTS, MAX_EVENT_BYTES), (MEMORY, MAX_MEMORY_BYTES), (CONVO, MAX_CONTEXT_CHARS * 4)):
        try:
            if target.is_file() and target.stat().st_size > limit:
                _atomic_write(target, target.read_text(encoding="utf-8", errors="replace")[-limit // 2:])
        except OSError: pass
    if JOBS_DIR.is_dir():
        for lp in JOBS_DIR.glob("*.log"):
            try:
                if lp.stat().st_size > MAX_JOB_LOG_BYTES:
                    _atomic_write(lp, lp.read_text(encoding="utf-8", errors="replace")[-MAX_JOB_LOG_BYTES // 2:])
            except OSError: pass
    jobs = state.setdefault("jobs", {})
    if len(jobs) > MAX_JOBS:
        finished = sorted((k for k,v in jobs.items() if v.get("status") != "running"), key=lambda k: jobs[k].get("started", ""))
        for jid in finished[:max(0, len(jobs)-MAX_JOBS)]: jobs.pop(jid, None)
    _mem_prune()


TOOLS = {
    "run_shell":      tool_run_shell,
    "run_background": tool_run_background,
    "check_jobs":     tool_check_jobs,
    "kill_job":       tool_kill_job,
    "read_file":      tool_read_file,
    "write_file":     tool_write_file,
    "append_file":    tool_append_file,
    "edit_file":      tool_edit_file,
    "list_dir":       tool_list_dir,
    "search_files":   tool_search_files,
    "fetch_url":      tool_fetch_url,
    "web_search":     tool_web_search,
    "update_notes":   tool_update_notes,
    "update_plan":    tool_update_plan,
    "git_checkpoint": tool_git_checkpoint,
    "restart_self":   tool_restart_self,
    "remember":       tool_remember,
    "recall":         tool_recall,
    "delegate":       tool_delegate,
}

for _tool_name, _tool_fn in list(TOOLS.items()):
    def _wrapped(args, _fn=_tool_fn, _name=_tool_name):
        _journal("intent", _name, args)
        try:
            _result = _fn(args)
        except Exception as _exc:
            _result = f"ERROR executing {_name}: {_exc}"
        _journal("result", _name, args, _result)
        return _result
    TOOLS[_tool_name] = _wrapped

TOOL_SPEC = """
You act by replying with EXACTLY ONE JSON object per turn — no prose, no markdown fences.

Actions:
  {"thought": "...", "action": "run_shell",      "args": {"command": "...", "timeout": 120}}
  {"thought": "...", "action": "run_background", "args": {"command": "..."}}
  {"thought": "...", "action": "check_jobs",     "args": {}}
  {"thought": "...", "action": "kill_job",       "args": {"job_id": "..."}}
  {"thought": "...", "action": "read_file",      "args": {"path": "...", "start_line": 1, "end_line": 200}}
  {"thought": "...", "action": "write_file",     "args": {"path": "...", "content": "..."}}
  {"thought": "...", "action": "append_file",    "args": {"path": "...", "content": "..."}}
  {"thought": "...", "action": "edit_file",      "args": {"path": "...", "old": "...", "new": "..."}}
  {"thought": "...", "action": "list_dir",       "args": {"path": ".", "depth": 3}}
  {"thought": "...", "action": "search_files",   "args": {"pattern": "regex", "path": ".", "ignore_case": true}}
  {"thought": "...", "action": "fetch_url",      "args": {"url": "..."}}
  {"thought": "...", "action": "web_search",     "args": {"query": "..."}}
  {"thought": "...", "action": "update_notes",   "args": {"markdown": "..."}}
  {"thought": "...", "action": "update_plan",    "args": {"steps": [{"step": "...", "status": "pending|in_progress|done"}]}}
  {"thought": "...", "action": "git_checkpoint", "args": {"message": "..."}}
  {"thought": "...", "action": "restart_self",   "args": {"reason": "..."}}
  {"thought": "...", "action": "remember",       "args": {"text": "durable fact/decision/gotcha", "kind": "fact|decision|gotcha|milestone"}}
  {"thought": "...", "action": "recall",         "args": {"query": "...", "top_k": 6}}
  {"thought": "...", "action": "delegate",       "args": {"task": "well-scoped subtask", "context": "what the worker needs to know", "max_turns": 40}}
  {"thought": "...", "action": "goal_complete",  "args": {"evidence": "...", "summary": "..."}}

Rules:
  - One action per turn. You receive the tool result, then act again.
  - Work incrementally. Verify every step with real command output before building on it.
  - Keep your plan current with update_plan and your memory current with update_notes.
    NOTES.md survives context compaction and restarts — the conversation does not.
  - Any command that may run longer than ~2 minutes MUST be launched with
    run_background and polled with check_jobs (read the job's log file for detail).
  - Use git_checkpoint after each working milestone so you can roll back mistakes.
  - Never repeat an approach that already failed; check the failure log first.
  - Only call goal_complete when you have CONCRETE, VERIFIED evidence (passing tests,
    working output) that the EXPECTATION is fully met. An independent verifier will
    re-run your checks — quote exactly what it should run to confirm.
  - You cannot declare impossibility. If stuck, change strategy: research with
    web_search/fetch_url, simplify, decompose, or build a verification harness and
    iterate against it.
  - Use remember for durable facts, decisions, and gotchas worth keeping for WEEKS —
    semantic memory survives compaction, restarts, and NOTES rotation. Use recall to
    search it before re-deriving something you may already know.
  - Use delegate for big, well-scoped subtasks (e.g. "research X and summarize",
    "build module Y with tests"). The worker gets fresh context and returns only a
    digest — your own context stays clean. Give it everything it needs in context.
"""


                                                                               
                                                             
                                                                               
def load_state() -> dict:
    if STATE.is_file():
        try:
            s = json.loads(STATE.read_text())
        except Exception:
            s = {}
    else:
        s = {}
    s.setdefault("goal", None)
    s.setdefault("expectation", None)
    s.setdefault("plan", [])
    s.setdefault("failures", [])
    s.setdefault("jobs", {})
    s.setdefault("iteration", 0)
    s.setdefault("segments", 0)
    s.setdefault("done", False)
    s.setdefault("stats", {"api_calls": 0, "est_tokens": 0, "started": None})
    return s


def save_state(state: dict) -> None:
    state["stats"]["api_calls"] = STATS["api_calls"]
    state["stats"]["est_tokens"] = STATS["est_tokens"]
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)                                               


def record_failure(state: dict, entry: str) -> None:
    if entry not in state["failures"]:
        state["failures"].append(entry)
        del state["failures"][:-50]
        save_state(state)


                                                                               
                                                                    
                                                                               
def save_conversation(messages, summary: str) -> None:
    tmp = CONVO.with_suffix(".tmp")
    tmp.write_text(json.dumps({"messages": messages, "summary": summary}), encoding="utf-8")
    tmp.replace(CONVO)


def load_conversation():
    if CONVO.is_file():
        try:
            data = json.loads(CONVO.read_text())
            if isinstance(data.get("messages"), list) and data["messages"]:
                return data
        except Exception:
            pass
    return None


def write_status(state: dict, last_action: str = "") -> None:
    payload = {
        "goal": state.get("goal"),
        "iteration": state.get("iteration"),
        "segments": state.get("segments"),
        "done": state.get("done"),
        "model": CURRENT_MODEL,
        "elapsed": fmt_elapsed(time.time() - START_T),
        "api_calls": STATS["api_calls"],
        "est_tokens": STATS["est_tokens"],
        "last_action": last_action or state.get("last_action", ""),
        "updated": iso(),
    }
    try:
        STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def webhook_ping(text: str) -> None:
    if not WEBHOOK_URL:
        return
    try:
        body = json.dumps({"content": text, "text": text}).encode()
        req = urllib.request.Request(WEBHOOK_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as exc:
        log.warning("webhook ping failed: %s", exc)


                                                                               
                                                                           
                                                                               
def context_size(messages) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def compact_context(messages, summary: str):
    """Summarize everything except the system prompt and the most recent turns."""
    if len(messages) <= KEEP_RECENT_TURNS + 2:
        return messages, summary
    old_chunk = messages[1:-KEEP_RECENT_TURNS]
    transcript = "\n".join(f"[{m['role']}] {m.get('content', '')[:1200]}" for m in old_chunk)
    transcript = transcript[-80_000:]

    compaction_prompt = [
        {"role": "system", "content": "You compress agent conversation history."},
        {"role": "user", "content": (
            "Existing running summary:\n" + (summary or "(none)") +
            "\n\nFold the following transcript chunk into the running summary. "
            "Preserve: the plan and which steps are done, decisions made, files "
            "created/modified, commands run and their key results, background jobs, "
            "current state of the work, and concrete next steps. "
            "Be dense and factual, under 600 words.\n\n" + transcript
        )},
    ]
    try:
        new_summary = llm_chat_power(compaction_prompt, max_tokens=1500)               
        if len(new_summary) > 12_000:
            new_summary = new_summary[-12_000:]
    except Exception as exc:
        log.error("Compaction failed (%s); dropping oldest turns instead.", exc)
        return [messages[0]] + messages[-KEEP_RECENT_TURNS:], summary

    log.info("Context compacted: %d turns folded into summary.", len(old_chunk))
    return [messages[0]] + messages[-KEEP_RECENT_TURNS:], new_summary


                                                                               
                                                                              
                                                                               
def extract_json_object(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
                                                                            
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
        start = text.find("{", start + 1)
    return None


def fingerprint(act: str, args: dict) -> str:
    try:
        blob = json.dumps([act, args], sort_keys=True, default=str)
    except Exception:
        blob = act + str(args)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


                                                                               
                                                               
                                                                               
VERIFIER_TOOLS = {"run_shell", "read_file", "list_dir", "search_files", "fetch_url", "check_jobs"}


def verify_completion(state: dict, evidence: str):
    sys_prompt = f"""You are an INDEPENDENT VERIFICATION AGENT on a Windows 11 machine (cwd: {WORKDIR}).
You are skeptical. You trust nothing you have not reproduced yourself.

GOAL:
{state['goal']}

EXPECTATION (definition of done — must be 100% satisfied):
{state['expectation']}

The working agent claims completion and offers this evidence:
{evidence}

Your job: prove or disprove the claim by RUNNING the acceptance checks yourself
(execute tests, builds, curls; read the produced artifacts). Do not trust quoted
output — reproduce it.

Reply with EXACTLY ONE JSON object per turn — no prose, no fences.
Available checks:
  {{"action": "run_shell",    "args": {{"command": "...", "timeout": 120}}}}
  {{"action": "read_file",    "args": {{"path": "...", "start_line": 1, "end_line": 200}}}}
  {{"action": "list_dir",     "args": {{"path": ".", "depth": 3}}}}
  {{"action": "search_files", "args": {{"pattern": "regex", "path": "."}}}}
  {{"action": "fetch_url",    "args": {{"url": "..."}}}}
  {{"action": "check_jobs",   "args": {{}}}}
When certain (within {VERIFIER_MAX_TURNS} turns), reply with a verdict:
  {{"action": "verdict", "verified": true,  "reason": "what you ran and saw"}}
  {{"action": "verdict", "verified": false, "gaps": "precisely what remains unverified or broken"}}"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Begin verification. Run the checks, then give your verdict."},
    ]
    for turn in range(VERIFIER_MAX_TURNS):
        heartbeat()
        try:
            reply = llm_chat_power(messages)                                            
        except Exception as exc:
            log.error("verifier model call failed: %s", exc)
            return False, f"verifier error: {exc}"
        action = extract_json_object(reply)
        if not action:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "Reply with exactly one JSON action object."})
            continue
        act = action.get("action")
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if act == "verdict":
            verified = bool(action.get("verified"))
            detail = action.get("reason") or action.get("gaps") or ""
            log.info("[verifier] verified=%s: %s", verified, str(detail)[:300])
            log_event("verifier", f"verified={verified}: {str(detail)[:300]}", remember=True)
            return verified, str(action.get("gaps") or action.get("reason") or "")
        if act not in VERIFIER_TOOLS:
            result = f"ERROR: verifier may only use: {', '.join(sorted(VERIFIER_TOOLS))}, or verdict."
        else:
            try:
                result = TOOLS[act](args)
            except Exception as exc:
                result = f"ERROR executing {act}: {exc}"
        log.info("[verifier] %s -> %.200s", act, result)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"CHECK RESULT ({act}):\n{result}"})
    return False, "verifier inconclusive within turn budget — treat as unverified"


                                                                               
                                                                           
                                                                               
def build_system_prompt(state: dict, summary: str) -> str:
    notes_excerpt = ""
    if NOTES.is_file():
        notes_excerpt = NOTES.read_text(encoding="utf-8", errors="replace")[-4000:]

    plan = state.get("plan") or []
    plan_text = "\n".join(
        "  [%s] %s" % ("x" if s.get("status") == "done" else "~" if s.get("status") == "in_progress" else " ",
                       s.get("step", ""))
        for s in plan
    ) or "  (no plan yet — create one with update_plan)"

    failures = "\n".join(f"  - {f}" for f in state.get("failures", [])[-15:]) or "  (none yet)"

    jobs = state.get("jobs") or {}
    running = [f"  - {jid}: {j.get('command', '')[:80]} (log: {j.get('log')})"
               for jid, j in jobs.items() if j.get("status") == "running"]
    jobs_text = "\n".join(running) or "  (none)"

                                                                            
    mem_block = ""
    try:
        focus = f"{state.get('goal', '')} {state.get('last_action', '')}"
        mem_block = memory_context_block(focus)
    except Exception as exc:
        log.warning("memory recall skipped: %s", exc)

                                                                           
    res_warn = resource_warnings()

    started = state.get("stats", {}).get("started")
    elapsed = fmt_elapsed(time.time() - START_T)

    return f"""You are an autonomous engineering agent running unattended on a Windows 11 machine.
You work until the GOAL is verifiably complete. You do not give up.

GOAL:
{state['goal']}

EXPECTATION (the definition of done — must be verifiably satisfied):
{state['expectation']}

ENVIRONMENT:
  - Windows 11 machine, headless. Working directory: {WORKDIR}
  - Python 3, common CLI tools available. Install packages with pip/winget as needed.
  - Current UTC time: {iso()}
  - Session elapsed: {elapsed} | iteration: {state.get('iteration')} (segment {state.get('segments', 0)})
  - API calls this run: {STATS['api_calls']} | est. tokens: ~{STATS['est_tokens']:,}
  - Mission started: {started or 'this session'}

CURRENT PLAN:
{plan_text}

RUNNING BACKGROUND JOBS:
{jobs_text}

RUNNING SUMMARY OF WORK SO FAR:
{summary or "(fresh start)"}

PERSISTENT NOTES (your long-term memory, most recent last):
{notes_excerpt or "(empty — use update_notes to build memory)"}

{mem_block}

KNOWN FAILED APPROACHES (do not repeat these):
{failures}

{("RESOURCE WARNINGS:\n" + res_warn) if res_warn else ""}

LONG-HORIZON OPERATING PRINCIPLES:
  - Decompose the mission into milestones; keep the plan above current with update_plan.
  - Verify everything with real command output. Never assume a step worked.
  - Long commands (builds, installs, training, servers) go to run_background; poll
    with check_jobs and read the job log file. Do not block on them.
  - git_checkpoint after every working milestone so you can diff and roll back.
  - Before context grows stale, write key facts to NOTES.md — it survives compaction.
  - If you improve this agent's own source code, call restart_self to adopt changes.

{TOOL_SPEC}"""


                                                                               
          
                                                                               
def _handle_signal(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True
    log.warning("Received signal %s — will save and exit after this iteration.", sig)


                                                                               
                  
                                                                               
PROGRESS_ACTIONS = {"write_file", "append_file", "edit_file", "update_plan",
                    "update_notes", "run_background", "git_checkpoint", "restart_self"}


def run_agent(cli_goal=None, cli_expectation=None) -> None:
    global AGENT_STATE, STATS
    state = load_state()
    AGENT_STATE = state
    STATS.update(state.get("stats", {}))

                                                                              
    if cli_goal:
        state["goal"] = cli_goal
    if cli_expectation:
        state["expectation"] = cli_expectation
    if not state["goal"]:
        print("=" * 70)
        print(" AUTONOMOUS GOAL AGENT  — NVIDIA NIM")
        print("=" * 70)
        state["goal"] = input("\nGOAL: ").strip()
        state["expectation"] = input("EXPECTATION: ").strip()
        if not state["goal"] or not state["expectation"]:
            print("Both GOAL and EXPECTATION are required.")
            sys.exit(1)
    else:
        log.info("Resuming. GOAL: %s", state["goal"])

    if state["done"]:
        log.info("Goal already marked complete. Run with --reset to start over.")
        return

    if not state["stats"].get("started"):
        state["stats"]["started"] = iso()
    state.setdefault("meta", {}).setdefault("run_started_epoch", time.time())
    save_state(state)
    start_watchdog()                              
    log_event("startup", f"agent started/resumed; goal: {str(state['goal'])[:200]}")

                                                      
    conv = load_conversation()
    if conv:
        messages = conv["messages"]
        summary = conv.get("summary", "")
        log.info("Resumed conversation checkpoint (%d turns).", len(messages))
    else:
        summary = ""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content":
                "Begin. Create your initial plan with update_plan, then start executing."},
        ]

    consecutive_parse_errors = 0
    last_fp = None
    fp_streak = 0
    last_progress_iter = state["iteration"]
    last_webhook = 0.0

    while True:
                                 
        if SHUTDOWN or RESTART_REQUESTED.is_set():
            save_conversation(messages, summary)
            save_state(state)
            write_status(state, "restart_requested" if RESTART_REQUESTED.is_set() else "shutdown")
            log.info("Checkpoint saved before exit.")
            return

                          
        if MAX_RUNTIME_DAYS and (time.time() - float(state.get("meta", {}).get("run_started_epoch", START_T))) > MAX_RUNTIME_DAYS * 86400:
            save_conversation(messages, summary)
            save_state(state)
            log.error("MAX_RUNTIME_DAYS reached. State saved.")
            return
        if MAX_API_CALLS and STATS["api_calls"] >= MAX_API_CALLS:
            save_conversation(messages, summary)
            save_state(state)
            log.error("MAX_API_CALLS reached. State saved.")
            return

        state["iteration"] += 1
        heartbeat()                             

                                                                  
        if state["iteration"] % MAINTENANCE_EVERY == 0:
            try:
                run_maintenance(state)
            except Exception as exc:
                log.warning("maintenance pass failed: %s", exc)

                                                                  
        if state["iteration"] > MAX_ITERATIONS:
            state["segments"] = state.get("segments", 0) + 1
            state["iteration"] = 1
            log.warning("Soft iteration cap hit — compacting and continuing (segment %d).",
                        state["segments"])
            messages, summary = compact_context(messages, summary)
            messages.append({"role": "user", "content":
                "Segment boundary reached. Review your plan and notes, then continue the mission."})

                                                          
        if context_size(messages) > MAX_CONTEXT_CHARS:
            messages, summary = compact_context(messages, summary)
            save_conversation(messages, summary)

                                                         
        messages[0] = {"role": "system", "content": build_system_prompt(state, summary)}
        CONV_REF["messages"] = messages
        CONV_REF["summary"] = summary

        save_state(state)
        write_status(state)
        log.info("── iteration %d (segment %d) ──", state["iteration"], state.get("segments", 0))

                                              
        if WEBHOOK_URL and time.time() - last_webhook > WEBHOOK_EVERY_S:
            last_webhook = time.time()
            webhook_ping(f"Agent iter {state['iteration']} | elapsed {fmt_elapsed(time.time() - START_T)} "
                         f"| goal: {str(state['goal'])[:120]}")

                                                                 
        stuck = fp_streak >= 3 or consecutive_parse_errors >= 2
        try:
            if stuck and POWER_MODEL:
                log.info("escalating turn to POWER_MODEL (stuck: streak=%d parse_errors=%d)",
                         fp_streak, consecutive_parse_errors)
                reply = llm_chat_power(messages)
            else:
                reply = llm_chat(messages)
        except Exception as exc:
            log.error("Model call failed hard: %s — retrying in 30s", exc)
            time.sleep(30)
            continue

        action = extract_json_object(reply)
        if action is None:
            action = extract_json_object(repair_json(reply))                             
            if action is not None:
                log.info("JSON repair salvaged a malformed reply")
        if action is None or "action" not in action:
            consecutive_parse_errors += 1
            log.warning("Unparseable reply (%d in a row): %.200s", consecutive_parse_errors, reply)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                "Your reply was not a valid single JSON action object. "
                "Reply with EXACTLY one JSON object per the action spec — no prose, no fences."})
            if consecutive_parse_errors >= 6:
                messages, summary = compact_context(messages, summary)
                messages = messages[:1] + [{"role": "user", "content":
                    "Resume the mission. Reply with exactly one JSON action object."}]
                consecutive_parse_errors = 0
            continue
        consecutive_parse_errors = 0

        act = str(action.get("action", ""))
        thought = str(action.get("thought", ""))
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        log.info("thought: %s", thought[:300])
        state["last_action"] = f"{act} — {thought[:120]}"

                                                                     
        fp = fingerprint(act, args)
        fp_streak = fp_streak + 1 if fp == last_fp else 1
        last_fp = fp
        loop_warning = ""
        if fp_streak >= 3:
            loop_warning = (f"\n\nWARNING: you have repeated this EXACT action {fp_streak} times "
                            f"in a row. If you are polling a background job, do other useful work "
                            f"or sleep between checks. If you are stuck, change strategy.")
            log.warning("repeat-loop detected: %s x%d", act, fp_streak)

                                                         
        if act == "goal_complete":
            evidence = str(args.get("evidence", ""))
            log.info("Agent claims completion. Evidence: %s", evidence[:500])
            verified, detail = verify_completion(state, evidence)
            if verified:
                state["done"] = True
                save_state(state)
                write_status(state, "goal_complete")
                save_conversation(messages, summary)
                if GIT_CHECKPOINTS:
                    tool_git_checkpoint({"message": "GOAL COMPLETE — final state"})
                webhook_ping(f"✔ GOAL COMPLETE: {str(args.get('summary', ''))[:300]}")
                log.info("✔ GOAL VERIFIED COMPLETE: %s", args.get("summary", ""))
                print("\n" + "=" * 70 + "\nGOAL COMPLETE\n" + "=" * 70)
                print(args.get("summary", ""))
                print("\nVerifier:", detail[:500])
                return
            log.info("Completion rejected: %s", detail[:300])
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                f"Completion NOT verified by the independent verifier. "
                f"Remaining gaps: {detail}\nContinue working."})
            continue

                              
        if act not in TOOLS:
            result = f"ERROR: unknown action '{act}'. Use one of: {', '.join(sorted(TOOLS))} or goal_complete."
        else:
            try:
                result = TOOLS[act](args)
            except KeyError as exc:
                result = f"ERROR: missing arg {exc} for action '{act}'"
            except Exception as exc:
                result = f"ERROR executing {act}: {exc}"

                                                                                       
        exit_code = None
        m = re.match(r"exit=(-?\d+)", result)
        if m:
            exit_code = int(m.group(1))
        if result.startswith("ERROR"):
            record_failure(state, f"iter {state['iteration']}: {act} "
                                  f"{json.dumps(args, default=str)[:150]} -> {result[:200]}")
            log_event("failure", f"iter {state['iteration']}: {act} -> {result[:200]}")
        elif act == "run_shell" and exit_code not in (0, None):
            fails = state.setdefault("exit_fails", {})
            fails[fp] = fails.get(fp, 0) + 1
            if len(fails) > 500:
                fails.clear()
            if exit_code >= 2 or fails[fp] >= 2:
                record_failure(state, f"iter {state['iteration']}: `{str(args.get('command', ''))[:120]}` "
                                      f"-> exit {exit_code}: {result[:150]}")
                log_event("failure", f"iter {state['iteration']}: shell exit {exit_code}: "
                                     f"{str(args.get('command', ''))[:120]}")

                                                              
        if act == "git_checkpoint" and "exit=0" in result:
            log_event("milestone", f"git checkpoint: {str(args.get('message', ''))[:200]}",
                      remember=True)

                                         
        if act in PROGRESS_ACTIONS:
            last_progress_iter = state["iteration"]
        stall_nudge = ""
        if state["iteration"] - last_progress_iter >= STALL_THRESHOLD:
            last_progress_iter = state["iteration"]
            stall_nudge = ("\n\nSTALL CHECK: many iterations have passed without any file change, "
                           "plan update, or notes update. Stop and reassess: are you making real "
                           "progress toward the EXPECTATION? Your next action should be update_plan "
                           "or update_notes recording your revised strategy.")
            log.warning("stall detected at iter %d", state["iteration"])
            log_event("stall", f"stall detected at iter {state['iteration']}", remember=True)

                                              
        reflection = ""
        if state["iteration"] % REFLECT_EVERY == 0:
            reflection = (f"\n\nREFLECTION CHECKPOINT (iteration {state['iteration']}, elapsed "
                          f"{fmt_elapsed(time.time() - START_T)}): your NEXT action should be "
                          f"update_plan (mark done steps, revise upcoming ones) or update_notes "
                          f"(record durable facts, decisions, and gotchas).")

        log.info("result: %.300s", result)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content":
            f"[iter {state['iteration']} | elapsed {fmt_elapsed(time.time() - START_T)}] "
            f"TOOL RESULT ({act}):\n{result}{loop_warning}{stall_nudge}{reflection}"})

                                                
        if state["iteration"] % CHECKPOINT_EVERY == 0:
            save_conversation(messages, summary)


                                                                               
                                                                       
                                                                               
def selftest() -> int:
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failures.append(name)
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print("=" * 60)
    print(" AUTONOMOUS GOAL AGENT  — offline self-test")
    print("=" * 60)

    def t_json_repair():
        broken = '{"thought": "x", "action": "run_shell", "args": {"command": "ls",}}'
        assert extract_json_object(repair_json(broken)) is not None, "trailing comma not repaired"
        broken2 = '{"thought": "line1\nline2", "action": "recall", "args": {"query": "q"'
        obj = extract_json_object(repair_json(broken2))
        assert obj and obj["action"] == "recall", "unclosed braces/newline not repaired"
    check("json repair", t_json_repair)

    def t_memory_keyword():
        EMBED_STATE["available"] = False                                     
        memory_store("The deploy server requires SSH key auth on port 2222, not password auth.", "gotcha")
        memory_store("Project deadline moved to next quarter after stakeholder review.", "fact")
        recs = memory_recall("how do I authenticate to the deploy server?")
        assert recs and "2222" in recs[0]["text"], f"keyword recall missed: {recs}"
        recs2 = memory_recall("when is the deadline?")
        assert recs2 and "quarter" in recs2[0]["text"], "keyword recall missed deadline"
    check("semantic memory (keyword fallback)", t_memory_keyword)

    def t_memory_persist():
        _MEM_CACHE["loaded"] = False
        _MEM_CACHE["items"] = []
        items = _mem_load()
        assert any("2222" in r["text"] for r in items), "memory did not persist to MEMORY.jsonl"
    check("memory persistence", t_memory_persist)

    def t_cosine():
        assert abs(_cosine([1, 0], [1, 0]) - 1.0) < 1e-9
        assert abs(_cosine([1, 0], [0, 1])) < 1e-9
        assert _cosine([0, 0], [1, 1]) == 0.0
    check("cosine similarity", t_cosine)

    def t_events():
        log_event("selftest", "self-test event")
        assert EVENTS.is_file() and "self-test event" in EVENTS.read_text(), "event not appended"
    check("event timeline", t_events)

    def t_resources():
        assert _disk_free_mb() > 0, "disk probe failed"
        assert _mem_free_mb() > 0, "mem probe failed"
        resource_warnings()                   
    check("resource governor probes", t_resources)

    def t_maintenance():
        run_maintenance({"iteration": 1, "meta": {}})                   
    check("maintenance pass", t_maintenance)

    def t_keypool():
        pool = KeyPool(["k1", "k2"])
        assert pool.get() == "k1"
        pool.report("k1", "rate")
        assert pool.get() == "k2", "rotation failed"
        pool.report("k1", "ok")
        pool2 = KeyPool([])
        assert pool2.get() == NVIDIA_API_KEY, "empty pool must fall back to single key"
    check("key rotation + circuit breaker", t_keypool)

    def t_heartbeat():
        heartbeat()
        assert HEARTBEAT.is_file(), "heartbeat file not written"
    check("watchdog heartbeat", t_heartbeat)

    def t_tools_registered():
        for name in ("remember", "recall", "delegate"):
            assert name in TOOLS, f"{name} missing from TOOLS"
        assert "remember" in TOOL_SPEC and "delegate" in TOOL_SPEC, "TOOL_SPEC missing  actions"
    check(" tools registered", t_tools_registered)

    def t_prompt_builds():
        s = load_state()
        s["goal"] = "deploy server authentication"                            
        s["expectation"] = "selftest expectation"
        p = build_system_prompt(s, "test summary")
        assert "deploy server authentication" in p, "prompt missing goal"
        assert "RECALLED LONG-TERM MEMORIES" in p, "prompt missing memory block"
        assert "2222" in p, "prompt missing recalled memory content"
    check("system prompt with memory block", t_prompt_builds)

    print("=" * 60)
    if failures:
        print(f" SELF-TEST FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print(" ALL CHECKS PASSED —  machinery is healthy (no API calls were made)")
    return 0


                                                                               
             
                                                                               
def main() -> None:
    global NVIDIA_API_KEY
    ap = argparse.ArgumentParser(description="Autonomous Goal Agent ")
    ap.add_argument("--goal", help="Mission goal (non-interactive start)")
    ap.add_argument("--expectation", help="Definition of done (non-interactive start)")
    ap.add_argument("--status", action="store_true", help="Print STATUS.json and exit")
    ap.add_argument("--reset", action="store_true", help="Clear STATE/CONVERSATION (keeps NOTES.md) and exit")
    ap.add_argument("--selftest", action="store_true", help="Validate  machinery offline (no API calls) and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.status:
        print(STATUS.read_text() if STATUS.is_file() else "(no STATUS.json — agent not running here)")
        return
    if args.reset:
        for f in (STATE, CONVO):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        print("State cleared (NOTES.md kept). Next run starts fresh.")
        return

    if not (NVIDIA_API_KEY or NVIDIA_API_KEYS):
        print("Set NVIDIA_API_KEY or NVIDIA_API_KEYS in the service environment.")
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        _acquire_lock()
        run_agent(cli_goal=args.goal or os.environ.get("AGENT_GOAL"), cli_expectation=args.expectation or os.environ.get("AGENT_EXPECTATION"))
    except KeyboardInterrupt:
        log.info("Interrupted. State saved — rerun to resume.")
    except Exception as exc:
        log.exception("Fatal agent error: %s", exc)
        with contextlib.suppress(Exception):
            if AGENT_STATE: save_state(AGENT_STATE)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
