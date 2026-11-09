"""HTTP + WebSocket server for the Neuron Lab UI.

    uv run python -m neuronlab.server --model Qwen/Qwen3.5-0.8B --port 8780

All GPU work is serialized by one lock. Generation takes it per token, so
neuron cards and searches stay responsive while a run is streaming.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import re
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import try_to_load_from_cache

from . import analysis, concepts, hunt
from .generate import GenConfig, Run, generate, sweep
from .metrics import Scorer
from .model import NeuronModel, Steer, normalize
from .stats import DATA, load_stats, model_dir

WEB = Path(__file__).resolve().parent.parent / "web"

MODELS = [
    {"repo": "Qwen/Qwen3.5-0.8B", "note": "default · 24 layers × 3584 · hybrid DeltaNet + attention"},
    {"repo": "Qwen/Qwen3-0.6B", "note": "simplest · 28 layers × 3072 · plain transformer"},
    {"repo": "Qwen/Qwen3.5-2B", "note": "bigger · hybrid · ~4.5 GB"},
    {"repo": "Qwen/Qwen3.5-4B", "note": "the evil-qwen post · 32 layers × 9216 · ~9 GB"},
]
PRESETS = [
    {"name": "evil qwen (lyra's neurons)", "model": "Qwen/Qwen3.5-4B",
     "steers": [{"layer": 14, "neuron": 8949, "value": 1.0, "mode": "dose"},
                {"layer": 13, "neuron": 7956, "value": -1.0, "mode": "dose"}]},
]
FRAME_UNIT = 32  # frames carry round(act / same-sign peak * 32) as int8, i.e. up to ±3.97x peak
ID_RE = re.compile(r"^\s*L?\s*(\d+)\s*(?:[\s:/,.\-]+|N)\s*N?\s*(\d+)\s*$", re.I)


def finite(x: float) -> float | None:
    return round(float(x), 5) if math.isfinite(float(x)) else None


def clean_json(x):
    """Non-finite floats become null, which JSON.parse accepts; numpy scalars become Python ones."""
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    return x


# A/B metrics: every score, acc (prompts with known answers only), and the ones useful steering lowers
AB_METRICS = Scorer.NAMES + ["acc", "acc1", "acc_real", "abstain_fake", "-words", "-markdown"]
ANSWER_METRICS = {"acc", "acc1", "acc_real", "abstain_fake"}  # need every prompt in concepts.ANSWERS


def found_neurons(repo: str) -> list[dict]:
    """Interventions validated by scripts/hunt.py for this model: specific finds first, then by p-value.
    Verdicts are recomputed from the saved rows (hunt.summarize), so every file is judged by the same rules."""
    out = []
    for path in sorted((DATA / "hunt" / repo.replace("/", "__")).glob("*.json")):
        try:
            res = json.loads(path.read_text())
            concept = res.get("concept") or path.stem.split("_pair_")[0]
            for v in res.get("validations") or ([res] if "steers" in res else []):
                verdict = hunt.summarize(v["rows"], v["metric"], v.get("guards"))
                if verdict["p"] is None:
                    continue
                out.append({"concept": concept, "about": concepts.CONCEPTS.get(concept, {}).get("about", ""),
                            "metric": v["metric"], "steers": v["steers"], "n_null": sum(r["kind"] == "null" for r in v["rows"]),
                            "guards": v.get("guards"), "guard": verdict.get("guard"), "guard_ok": verdict.get("guard_ok"),
                            "useful": verdict.get("useful"),
                            **{k: verdict[k] for k in ("effect", "z", "p", "percentile", "null_mean", "null_std",
                                                       "null_max", "coherent", "specific")}})
        except Exception as e:  # one odd file must not break loading
            print(f"skipping {path.name}: {type(e).__name__}: {e}", flush=True)
    return sorted(out, key=lambda f: (not f["specific"], f["p"], -f["effect"]))


class Lab:
    def __init__(self):
        self.nm: NeuronModel | None = None
        self.vocab: analysis.Vocab | None = None
        self.stats = None
        self.scales_cpu: torch.Tensor | None = None
        self.resid_typical: list[float] = []
        self.mass: np.ndarray | None = None
        self.mass_rank: np.ndarray | None = None
        self.top_tok: np.ndarray | None = None
        self.run: Run | None = None
        self.labels: dict[str, str] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.examples_cache: OrderedDict = OrderedDict()
        self.dom_neg: np.ndarray | None = None  # [L, D] neurons that mostly fire negative
        self.jobs: set = set()  # (stop event, future) of streaming jobs
        self.scorer: Scorer | None = None  # text classifiers for A/B tests, loaded on first use
        self.found: list[dict] = []  # validated finds from scripts/hunt.py for the loaded model
        self.status = "starting"

    def load(self, repo: str):
        # stop streaming jobs first: they take the lock per token and hold the old model
        for stop, fut in list(self.jobs):
            stop.set()
        for stop, fut in list(self.jobs):
            try:
                fut.result(timeout=60)
            except Exception:
                pass
        with self.lock:
            prev = self.nm.repo if self.nm is not None else None
            old, self.nm, self.run = self.nm, None, None
            self.status = f"loading {repo}"
            if old is not None:
                old.unload()
                del old
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            try:
                self._build(repo)
            except Exception as e:
                print(f"loading {repo} failed: {e}", flush=True)
                if prev is None:
                    self.status = f"load failed: {e}"
                    raise
                self._build(prev)  # fall back to the model that was loaded before
                raise RuntimeError(f"could not load {repo} ({e}); kept {prev}") from e

    def _build(self, repo: str):
        """Load `repo` and everything derived from it. Caller holds the lock."""
        t0 = time.time()
        nm = NeuronModel(repo)
        self.vocab = analysis.Vocab(nm)
        self.stats = load_stats(repo)
        self.mass = self.mass_rank = self.top_tok = None
        if self.stats is not None:
            nm.scales = self.stats.scales().to(nm.device)
            mass = self.stats["mean_abs"] * nm.w_out_norm.cpu().numpy()
            order = np.argsort(-mass, axis=None)
            rank = np.empty(mass.size, dtype=np.int64)
            rank[order] = np.arange(1, mass.size + 1)
            self.mass_rank = rank.reshape(mass.shape)
            self.mass = mass
            flat = self.stats.tokens.reshape(-1)
            self.top_tok = flat[self.stats["top_p"]]  # [L, K, D] token at each top example
        self.scales_cpu = nm.scales.float().cpu()
        self.dom_neg = (self.scales_cpu[1] > self.scales_cpu[0]).numpy()
        self.resid_typical = [round(float(x), 2) for x in nm.resid_typical.cpu()]  # read by /api/info without the lock
        labels_path = model_dir(repo) / "labels.json"
        self.labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        self.found = found_neurons(repo)
        self.examples_cache.clear()
        self.run = None
        self.nm = nm
        self.status = "ready"
        print(f"loaded {repo} in {time.time() - t0:.1f}s (stats: {'yes' if self.stats else 'no'})", flush=True)

    def info(self) -> dict:
        nm = self.nm
        models = []
        for m in MODELS:
            models.append({**m, "cached": isinstance(try_to_load_from_cache(m["repo"], "config.json"), str),
                           "stats": (model_dir(m["repo"]) / "stats.npz").exists()})
        return {
            "status": self.status,
            "model": nm.repo if nm else None,
            "n_layers": nm.n_layers if nm else 0,
            "d_mlp": nm.d_mlp if nm else 0,
            "d_model": nm.d_model if nm else 0,
            "vocab": nm.vocab if nm else 0,
            "layer_types": nm.layer_types if nm else [],
            "device": nm.device if nm else None,
            "stats": self.stats.meta if self.stats else None,
            "scale_source": "corpus" if self.stats else "calibration",
            "resid_typical": self.resid_typical if nm else [],
            "models": models,
            "presets": PRESETS + [
                {"name": f"{f['concept']} · " + " ".join(f"L{s['layer']} N{s['neuron']} {s['value']:+g}" for s in f["steers"])
                         + " (found)", "model": nm.repo, "steers": f["steers"]}
                for f in self.found if nm and f["specific"]],
        }

    # ---- decorations -------------------------------------------------------

    def lens_word(self, layer: int, n: int, negative: bool | None = None) -> str:
        """Top token the neuron writes when firing with the given sign (default: its usual sign)."""
        if self.stats is None:
            return ""
        if negative is None:
            negative = bool(self.dom_neg[layer, n])
        return self.vocab.strs[int(self.stats["lens_minus" if negative else "lens_plus"][layer, n, 0])]

    def eff_lens(self) -> np.ndarray:
        """[L, D, k] lens tokens for each neuron's usual firing sign."""
        return np.where(self.dom_neg[:, :, None], self.stats["lens_minus"], self.stats["lens_plus"])

    def tag(self, item: dict) -> dict:
        L, n = item["layer"], item["neuron"]
        item["label"] = self.labels.get(f"{L}:{n}", "")
        rel = item.get("rel")
        item["lens"] = self.lens_word(L, n, None if rel is None else rel < 0)
        return item


lab = Lab()
app = FastAPI(title="Neuron Lab")


LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def is_local(host: str | None, origin: str | None) -> bool:
    """Only this machine may drive the model: blocks DNS-rebinding and cross-site requests."""
    if not host or host.rsplit(":", 1)[0] not in LOCAL_HOSTS:
        return False
    if origin:
        o = origin.split("://", 1)[-1].rsplit(":", 1)[0]
        return o in LOCAL_HOSTS
    return True


@app.middleware("http")
async def no_cache(request: Request, call_next):
    if not is_local(request.headers.get("host"), request.headers.get("origin")):
        return JSONResponse({"detail": "local requests only"}, status_code=403)
    if request.url.path.startswith("/api/"):
        # Requiring a custom header forces a CORS preflight that other sites cannot pass, so even
        # no-cors GETs (which carry no Origin) from a page in the same browser cannot drive the model.
        site = request.headers.get("sec-fetch-site")
        if request.headers.get("x-nl") != "1" or site not in (None, "same-origin", "none"):
            return JSONResponse({"detail": "API requests need the X-NL: 1 header"}, status_code=403)
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


async def gpu(fn, *args):
    """Run fn in a worker thread while holding the GPU lock."""

    def locked():
        with lab.lock:
            return fn(*args)

    return await asyncio.get_running_loop().run_in_executor(lab.pool, locked)


def need_run(run_id: int | None) -> Run:
    """The current run, refusing stale requests from a tab that is looking at an older one."""
    run = lab.run
    if run is None:
        raise HTTPException(404, "no run yet")
    if run_id is not None and run_id != run.id:
        raise HTTPException(409, "that run was replaced by a newer one")
    return run


def need_model() -> NeuronModel:
    if lab.nm is None:
        raise HTTPException(503, lab.status)
    return lab.nm


def check_neuron(layer: int, n: int):
    nm = need_model()
    if not (0 <= layer < nm.n_layers and 0 <= n < nm.d_mlp):
        raise HTTPException(404, f"L{layer} N{n} is out of range for {nm.repo}")


# ---- REST ------------------------------------------------------------------


@app.get("/api/info")
async def api_info():
    return lab.info()


@app.post("/api/load")
async def api_load(body: dict):
    repo = body["repo"]
    try:
        await asyncio.get_running_loop().run_in_executor(lab.pool, lab.load, repo)
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return lab.info()


@app.get("/api/neuron")
async def api_neuron(layer: int, n: int, clean: int = 1):
    check_neuron(layer, n)
    nm = lab.nm

    def work():
        card = {
            "layer": layer, "neuron": n, "label": lab.labels.get(f"{layer}:{n}", ""),
            "layer_type": nm.layer_types[layer],
            "w_out_norm": finite(nm.w_out_norm[layer, n]),
            "scale_pos": finite(lab.scales_cpu[0, layer, n]),
            "scale_neg": finite(lab.scales_cpu[1, layer, n]),
            "lens": analysis.logit_lens(nm, lab.vocab, layer, n, clean=bool(clean)),
            "web": analysis.web(nm, layer, n),
        }
        if lab.stats is not None:
            st = lab.stats
            card["stats"] = {k: finite(st[k][layer, n]) for k in ("max", "min", "mean", "mean_abs", "freq_pos", "freq_neg")}
            card["stats"]["mass"] = finite(lab.mass[layer, n])
            card["stats"]["mass_rank"] = int(lab.mass_rank[layer, n])
            card["stats"]["n_neurons"] = int(lab.mass.size)
        for side in ("up", "down"):
            for key in ("pos", "neg"):
                card["web"][side][key] = [lab.tag(x) for x in card["web"][side][key]]
        return card

    return await gpu(work)


@app.post("/api/steer_lens")
async def api_steer_lens(body: dict):
    layer, n = int(body["layer"]), int(body["neuron"])
    check_neuron(layer, n)
    return await gpu(analysis.steer_lens, lab.nm, lab.vocab, layer, n, float(body.get("dose", 1.0)), 12,
                     bool(body.get("clean", True)))


@app.get("/api/examples")
async def api_examples(layer: int, n: int, k: int = 10, neg: int = 0):
    check_neuron(layer, n)
    if lab.stats is None:
        return {"examples": [], "note": "no corpus stats yet — run python -m neuronlab.stats"}
    key = (lab.nm.repo, layer, n, k, neg)
    if key in lab.examples_cache:
        return lab.examples_cache[key]
    st, nm = lab.stats, lab.nm
    vals = st["bot_v" if neg else "top_v"][layer, :, n].astype(np.float32)
    poss = st["bot_p" if neg else "top_p"][layer, :, n]
    order = np.argsort(vals) if neg else np.argsort(-vals)
    T = st.tokens.shape[1]
    picks = [(int(poss[i]), float(vals[i])) for i in order if np.isfinite(vals[i])][:k]
    seqs, spots = [], []
    for flat, v in picks:
        s, t = divmod(flat, T)
        end = min(t + 7, int(st.lengths[s]))
        seqs.append(st.tokens[s, :end].tolist())
        spots.append((s, t, v))
    acts = await gpu(analysis.token_acts, nm, seqs, layer, n) if seqs else []
    out = []
    for seq, a, (s, t, v) in zip(seqs, acts, spots):
        lo = max(0, t - 28)
        out.append({
            "source": st.meta["sources"][int(st.source[s])],
            "value": finite(v),
            "peak": t - lo,
            "tokens": nm.token_strs(seq[lo:]),
            "acts": [finite(x) for x in a[lo:]],
            "rel": rel_list(a[lo:], layer, n),
        })
    res = {"examples": out}
    lab.examples_cache[key] = res
    if len(lab.examples_cache) > 256:
        lab.examples_cache.popitem(last=False)
    return res


@app.post("/api/probe_neuron")
async def api_probe_neuron(body: dict):
    """One neuron's activation on every token of a custom text."""
    layer, n = int(body["layer"]), int(body["neuron"])
    check_neuron(layer, n)
    nm = lab.nm
    ids = nm.chat_ids([{"role": "user", "content": body["text"]}]) if body.get("chat") else nm.text_ids(body["text"])
    acts = (await gpu(analysis.token_acts, nm, [ids], layer, n))[0]
    return {"tokens": nm.token_strs(ids), "acts": [finite(x) for x in acts], "rel": rel_list(acts, layer, n)}


def rel_list(acts: list[float], layer: int, n: int) -> list[float | None]:
    """Activations relative to the neuron's own peak of the same sign."""
    sp, sn = float(lab.scales_cpu[0, layer, n]), float(lab.scales_cpu[1, layer, n])
    return [finite(a / sp if a >= 0 else a / sn) for a in acts]


def encode_frame(pos: int, acts: torch.Tensor) -> bytes:
    """[rows, L, D] -> header (kind, pos, rows, L, D as int32) + int8 of act / same-sign peak * FRAME_UNIT."""
    q = normalize(acts, lab.scales_cpu).mul_(FRAME_UNIT).clamp_(-127, 127).round_().to(torch.int8)
    return struct.pack("<5i", 1, pos, *q.shape) + q.numpy().tobytes()


@app.get("/api/run/frame")
async def api_run_frame(pos: int, run: int | None = None):
    run = need_run(run)
    if not (0 <= pos < len(run.acts)):
        raise HTTPException(404, "no such position")
    return Response(encode_frame(pos, run.acts[pos]), media_type="application/octet-stream")


@app.get("/api/run/top")
async def api_run_top(pos: int, row: int = 0, k: int = 40, sign: str = "abs", diff: int = 0, run: int | None = None):
    run = need_run(run)
    if not (0 <= pos < len(run.acts)):
        raise HTTPException(404, "no such position")
    frame = run.acts[pos]
    if diff and frame.shape[0] >= 2:
        items = analysis.top_at(lab.nm, frame[1], lab.scales_cpu, k=k, sign=sign, base=frame[0])
    else:
        items = analysis.top_at(lab.nm, frame[min(row, frame.shape[0] - 1)], lab.scales_cpu, k=k, sign=sign)
    return {"pos": pos, "top": [lab.tag(x) for x in items]}


@app.get("/api/run/neuron")
async def api_run_neuron(layer: int, n: int, run: int | None = None):
    run = need_run(run)
    check_neuron(layer, n)
    vals = torch.stack([a[:, layer, n] for a in run.acts], dim=1).float()  # [rows, positions]
    return {"rows": [[finite(x) for x in r] for r in vals.tolist()],
            "rel": [rel_list(r, layer, n) for r in vals.tolist()]}


@app.post("/api/explain")
async def api_explain(body: dict):
    """Which neurons drove the token predicted at `pos` of a run row (attribution patching)."""
    nm = need_model()
    run = need_run(body.get("run"))
    row = min(int(body.get("row", 0)), len(run.rows) - 1)
    pos = int(body["pos"])
    toks = run.tokens[row]
    if not (0 <= pos < len(toks) - 1) and body.get("target") is None:
        raise HTTPException(400, "no next token at that position")
    target = int(body["target"]) if body.get("target") is not None else toks[pos + 1]
    steers = run.steers if run.rows[row] != "baseline" else []

    def work():
        add, clamp = nm.build_rows(steers, [run.rows[row]], run.resid_median) if steers else ({}, {})
        if steers and not run.steer_prompt:
            # replay the run faithfully: steering only touched generated positions
            gen = (torch.arange(pos + 1, device=nm.device) >= run.n_prompt)[None, :, None]
            add = {k: v * gen for k, v in add.items()}
            clamp = {k: (m & gen, v) for k, (m, v) in clamp.items()}
        return analysis.explain(nm, toks[: pos + 1], target, add, clamp)

    res = await gpu(work)
    for key in ("support", "oppose"):
        res[key] = [lab.tag(x) for x in res[key]]
    res.update({"pos": pos, "row": run.rows[row], "target": {"id": target, "text": nm.tok.decode([target])}})
    return res


@app.get("/api/run/lens")
async def api_run_lens(row: int = 0, run: int | None = None):
    """Logit-lens grid for one row of the current run (replayed with the run's steering)."""
    nm = need_model()
    run = need_run(run)
    row = min(row, len(run.rows) - 1)
    toks = run.tokens[row]
    steers = run.steers if run.rows[row] != "baseline" else []

    def work():
        add, clamp = nm.build_rows(steers, [run.rows[row]], run.resid_median) if steers else ({}, {})
        if steers and not run.steer_prompt:
            gen = (torch.arange(len(toks), device=nm.device) >= run.n_prompt)[None, :, None]
            add = {k: v * gen for k, v in add.items()}
            clamp = {k: (m & gen, v) for k, (m, v) in clamp.items()}
        return analysis.lens_grid(nm, lab.vocab, toks, add, clamp)

    grid = await gpu(work)
    words = {i for layer in grid for pos in layer for i, _ in pos}
    return {"row": run.rows[row], "n_prompt": run.n_prompt, "tokens": nm.token_strs(toks),
            "ids": toks, "grid": grid, "words": {i: lab.vocab.strs[i] for i in words}}


@app.post("/api/sweep")
async def api_sweep(body: dict):
    """Scale the given interventions by several factors and generate a reply for each."""
    nm = need_model()
    fields = GenConfig.__dataclass_fields__
    cfg = GenConfig(**{k: v for k, v in body.get("cfg", {}).items() if k in fields})
    steers = parse_steers(body.get("steers", []))
    if not steers:
        raise HTTPException(400, "add an intervention first")
    factors = [float(f) for f in body.get("factors", [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2])][:12]
    system = body.get("system", "").strip()
    messages = ([{"role": "system", "content": system}] if system else []) + body["messages"]
    ids = nm.chat_ids(messages, thinking=cfg.thinking)
    rows = await asyncio.get_running_loop().run_in_executor(lab.pool, sweep, nm, ids, steers, factors, cfg, lab.lock)
    return {"rows": rows}


@app.get("/api/search")
async def api_search(q: str, k: int = 24):
    nm = need_model()
    q = q.strip()
    m = ID_RE.match(q)
    if m:
        L, n = int(m.group(1)), int(m.group(2))
        ok = 0 <= L < nm.n_layers and 0 <= n < nm.d_mlp
        return {"id": {"layer": L, "neuron": n, "valid": ok}, "lens": [], "fires": [], "labels": []}
    ql = q.lower()
    labels = [lab.tag({"layer": int(key.split(":")[0]), "neuron": int(key.split(":")[1])})
              for key, v in lab.labels.items() if ql in v.lower()][:k]
    out = {"id": None, "labels": labels, "lens": [], "fires": []}
    if lab.stats is None or len(ql) < 2:
        return out
    match = np.array([s.strip().lower().startswith(ql) for s in lab.vocab.strs])
    ids = np.nonzero(match)[0]
    if ids.size == 0:
        return out
    lens = lab.eff_lens()  # [L, D, 8], for each neuron's usual firing sign
    hit = np.isin(lens, ids)
    weight = (np.arange(lens.shape[2], 0, -1))[None, None, :]
    score = (hit * weight).sum(-1).astype(np.float32)
    for i in np.argsort(-score, axis=None)[:k]:
        L, n = divmod(int(i), nm.d_mlp)
        if score[L, n] <= 0:
            break
        out["lens"].append(lab.tag({"layer": L, "neuron": n, "score": float(score[L, n])}))
    fires = np.isin(lab.top_tok, ids).sum(1).astype(np.float32)  # [L, D] how many top examples peak on it
    for i in np.argsort(-fires, axis=None)[:k]:
        L, n = divmod(int(i), nm.d_mlp)
        if fires[L, n] <= 0:
            break
        out["fires"].append(lab.tag({"layer": L, "neuron": n, "score": float(fires[L, n])}))
    return out


@app.post("/api/label")
async def api_label(body: dict):
    nm = need_model()
    key = f"{int(body['layer'])}:{int(body['neuron'])}"
    text = str(body.get("label", "")).strip()
    if text:
        lab.labels[key] = text
    else:
        lab.labels.pop(key, None)
    path = model_dir(nm.repo)
    path.mkdir(parents=True, exist_ok=True)
    (path / "labels.json").write_text(json.dumps(lab.labels, indent=1, ensure_ascii=False))
    return {"ok": True, "labels": len(lab.labels)}


@app.get("/api/lens_words")
async def api_lens_words():
    """Top promoted plain-word token of every neuron, flattened [layer * d_mlp + n]."""
    if lab.stats is None:
        return {"words": []}
    first = lab.eff_lens()[:, :, 0].reshape(-1)
    return {"words": [lab.vocab.strs[int(i)].strip() for i in first]}


@app.get("/api/labels")
async def api_labels():
    return {"labels": [lab.tag({"layer": int(k.split(":")[0]), "neuron": int(k.split(":")[1])})
                       for k in lab.labels]}


# ---- hunting ---------------------------------------------------------------


@app.get("/api/hunt")
async def api_hunt():
    """Concepts with contrast data, A/B metrics, default A/B prompts, and validated finds."""
    nm = need_model()
    lab.found = found_neurons(nm.repo)  # re-read, so finds saved by scripts/hunt.py show up without a restart
    return {"concepts": [{"name": k, "kind": c["kind"], "metric": c["metric"], "about": c["about"], "guards": c.get("guards"),
                          "prompts": c.get("validate"), "tokens": c.get("tokens")} for k, c in concepts.CONCEPTS.items()],
            "metrics": AB_METRICS, "prompts": concepts.SCREEN, "found": lab.found}


@app.post("/api/hunt/scan")
async def api_hunt_scan(body: dict):
    """Candidate neurons for a concept: the read scan (fires differently on concept text) and, for
    reply concepts, the first-order write scan (a +1 dose raises log p of concept replies)."""
    nm = need_model()
    k = max(1, min(int(body.get("k", 16)), 64))
    name = body.get("concept")
    if name:
        if name not in concepts.CONCEPTS:
            raise HTTPException(404, f"unknown concept {name}")
    else:
        pos = [t.strip()[:2000] for t in body.get("pos", []) if t.strip()][:64]
        neg = [t.strip()[:2000] for t in body.get("neg", []) if t.strip()][:64]
        if len(pos) < 2 or len(neg) < 2:
            raise HTTPException(400, "give at least two texts on each side")
        short = [t for t in pos + neg if len(nm.text_ids(t)) < 2]
        if short:
            raise HTTPException(400, f"too short to read (needs 2+ tokens): {', '.join(map(repr, short[:3]))}")

    def work():
        if name:
            read = hunt.read_scan(nm, name)
            write = hunt.steer_grad(nm, name) if concepts.CONCEPTS[name]["kind"] != "request" else None
        else:
            read, write = hunt.read_scan_texts(nm, pos, neg), None
        out = {"read": [], "write": []}
        for c in hunt.top_cands(read["t"], k, read["delta"].abs() >= 0.05):
            out["read"].append({**c.as_dict(), "t": finite(read["t"][c.layer, c.neuron]),
                                "delta": finite(read["delta"][c.layer, c.neuron]),
                                "agree": finite(read["agree"][c.layer, c.neuron])})
        if write is not None:
            for c in hunt.top_cands(write, k):
                out["write"].append({**c.as_dict(), "first_order": finite(write[c.layer, c.neuron] * c.sign)})
        return out

    out = await gpu(work)
    for item in out["read"] + out["write"]:
        item["label"] = lab.labels.get(f"{item['layer']}:{item['neuron']}", "")
        item["lens"] = lab.lens_word(item["layer"], item["neuron"], item["sign"] < 0)
    return out


# ---- streaming -------------------------------------------------------------


def parse_steers(raw: list[dict]) -> list[Steer]:
    return [Steer(int(s["layer"]), int(s["neuron"]), float(s["value"]), s.get("mode", "dose"))
            for s in raw if s.get("enabled", True)]


def run_job(msg: dict, emit, stop: threading.Event):
    nm = lab.nm
    try:
        if nm is None:
            raise RuntimeError(lab.status)
        fields = GenConfig.__dataclass_fields__
        cfg = GenConfig(**{k: v for k, v in msg.get("cfg", {}).items() if k in fields})
        steers = parse_steers(msg.get("steers", []))
        if msg["type"] == "probe":
            cfg.max_new_tokens = 0
            text = msg["text"]
            ids = nm.chat_ids([{"role": "user", "content": text}], thinking=cfg.thinking) if msg.get("chat") else nm.text_ids(text)
        else:
            system = msg.get("system", "").strip()
            messages = ([{"role": "system", "content": system}] if system else []) + msg["messages"]
            ids = nm.chat_ids(messages, thinking=cfg.thinking)

        def emit_run(kind, data):
            if kind == "start":
                lab.run = data.pop("_run")
            emit(kind, data)

        generate(nm, ids, steers, cfg, emit_run, stop, lab.lock)
    except Exception as e:  # report to the UI instead of dying silently
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"{type(e).__name__}: {e}"})


def ab_job(msg: dict, emit, stop: threading.Event):
    """A/B test interventions on several prompts (greedy): baseline, a dose curve, and random neurons
    of the same layers at the same doses as a null. Emits ab_progress, then ab_done."""
    nm = lab.nm
    try:
        if nm is None:
            raise RuntimeError(lab.status)
        steers = parse_steers(msg.get("steers", []))
        if not steers:
            raise RuntimeError("add an intervention first (Steer tab)")
        metric = msg.get("metric", "malice")
        if metric not in AB_METRICS:
            raise RuntimeError(f"unknown metric {metric}")
        prompts = [p.strip() for p in msg.get("prompts", []) if p.strip()][:24] or concepts.SCREEN
        if hunt.metric_key(metric)[0] in ANSWER_METRICS and not all(p in concepts.ANSWERS for p in prompts):
            raise RuntimeError(f"{metric} needs prompts with known answers: pick brief, plaintext or honest in the Hunt list")
        guards = msg.get("guards") or None
        curve = tuple(float(f) for f in msg.get("curve", [0.5, 1.0, 1.5]))[:6]
        n_null = max(0, min(int(msg.get("n_null", 24)), 48))
        cfg = GenConfig(temperature=0.0, max_new_tokens=max(8, min(int(msg.get("max_new_tokens", 64)), 160)),
                        steer_prompt=bool(msg.get("steer_prompt", True)))
        if lab.scorer is None:
            emit("ab_progress", {"done": 0, "total": len(prompts), "note": "loading the metric classifiers…"})
            scorer = Scorer()
            scorer.emotion, scorer.toxicity  # load both now rather than after the generations
            lab.scorer = scorer
        emit("ab_progress", {"done": 0, "total": len(prompts)})
        if guards and not all(p in concepts.ANSWERS for p in prompts):
            guards = {g: v for g, v in guards.items() if hunt.metric_key(g)[0] not in ANSWER_METRICS} or None  # no answers
        v = hunt.validate(nm, lab.scorer, steers, metric, prompts=prompts, n_null=n_null, curve=curve, cfg=cfg,
                          lock=lab.lock, stop=stop, progress=lambda d, t: emit("ab_progress", {"done": d, "total": t}),
                          guards=guards)
        for r in v["rows"]:
            r.pop("scores", None)
            if r["kind"] == "null":
                r.pop("texts", None)
        emit("ab_done", {"result": clean_json(v), "prompts": prompts})
    except hunt.Stopped:
        emit("ab_done", {"stopped": True})
    except Exception as e:
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"{type(e).__name__}: {e}"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not is_local(ws.headers.get("host"), ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    job = None

    def emit(kind, data):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    async def sender():
        while True:
            kind, data = await queue.get()
            if kind == "frame":
                await ws.send_bytes(encode_frame(data["pos"], data["acts"]))
            else:
                await ws.send_text(json.dumps({"type": kind, **data}))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "stop":
                stop.set()
            elif msg.get("type") in ("generate", "probe", "ab"):
                if job is not None and not job.done():
                    stop.set()
                    await job
                stop = threading.Event()
                cf = lab.pool.submit(ab_job if msg["type"] == "ab" else run_job, msg, emit, stop)
                entry = (stop, cf)
                lab.jobs.add(entry)
                cf.add_done_callback(lambda _f, e=entry: lab.jobs.discard(e))
                job = asyncio.wrap_future(cf)
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        send_task.cancel()


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    args = ap.parse_args()
    lab.load(args.model)
    print(f"Neuron Lab on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
