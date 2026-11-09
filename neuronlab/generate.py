"""Streaming generation with an optional baseline-vs-steered comparison.

Rows of one batch share the prompt and the sampling noise (Gumbel-max with
one noise vector per step), so with no steering they produce identical text
and any divergence is caused by the steering applied to the "steered" row.
"""
from __future__ import annotations

import itertools
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable

import torch

from .model import NeuronModel, Steer, merge_steering, scaled


@dataclass
class GenConfig:
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: int = 0
    thinking: bool = False
    compare: bool = True
    inverse: bool = False  # with compare, add a row where every dose/add is negated
    steer_prompt: bool = True  # steer prompt positions too ("at every input"), not just generated ones
    n_alts: int = 5


_run_ids = itertools.count(1)


@dataclass
class Run:
    """The latest generation or probe, kept so the UI can query any position later."""

    rows: list[str]
    tokens: list[list[int]]  # per row: prompt + generated ids (a finished row stops growing)
    n_prompt: int
    acts: list[torch.Tensor] = field(default_factory=list)  # per position: [rows, n_layers, d_mlp] bf16, CPU
    steers: list[Steer] = field(default_factory=list)
    resid_median: torch.Tensor | None = None
    steer_prompt: bool = True
    kl: list[float | None] = field(default_factory=list)  # per generated position
    id: int = field(default_factory=lambda: next(_run_ids))


def sample(logits: torch.Tensor, cfg: GenConfig, gen: torch.Generator, history: list[list[int]]) -> torch.Tensor:
    """Pick one token per row. All rows share the same Gumbel noise."""
    logits = logits.clone()
    if cfg.repetition_penalty != 1.0:
        for r, hist in enumerate(history):
            idx = torch.tensor(sorted(set(hist)), device=logits.device)
            vals = logits[r, idx]
            logits[r, idx] = torch.where(vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty)
    if cfg.temperature <= 0:
        return logits.argmax(-1)
    logits = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = logits.topk(min(cfg.top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p < 1.0:
        sorted_l, order = logits.sort(dim=-1, descending=True)
        probs = sorted_l.softmax(-1)
        drop = probs.cumsum(-1) - probs > cfg.top_p  # always keeps the most likely token
        sorted_l = sorted_l.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_l)
    u = torch.rand(logits.shape[-1], generator=gen).clamp_(1e-10, 1 - 1e-10).to(logits.device)
    return (logits - torch.log(-torch.log(u))).argmax(-1)


@torch.no_grad()
def generate(
    nm: NeuronModel,
    ids: list[int],
    steers: list[Steer],
    cfg: GenConfig,
    emit: Callable[[str, dict], None],
    stop: threading.Event,
    lock: threading.Lock,
) -> Run:
    """Run the prompt (and generate up to cfg.max_new_tokens), emitting events:

    start  {rows, n_prompt, tokens, offsets}
    frame  {pos, acts [rows, n_layers, d_mlp] bf16}   one per position
    token  {pos, rows: [{id, text, p, alts, done}], kl}
    done   {reason, n_generated, seconds, tok_per_s}
    """
    dev = nm.device
    if steers and cfg.compare:
        rows = ["baseline", "steered"] + (["inverse"] if cfg.inverse else [])
    elif steers:
        rows = ["steered"]
    else:
        rows = ["baseline"]
    B = len(rows)

    with lock:
        prompt = torch.tensor([ids], device=dev)
        resid_median = nm.resid_median(prompt) if any(s.mode == "dose" for s in steers) else None
        add, clamp = nm.build_rows(steers, rows, resid_median) if steers else ({}, {})
        run = Run(rows=rows, tokens=[list(ids) for _ in rows], n_prompt=len(ids), steers=steers,
                  resid_median=resid_median, steer_prompt=cfg.steer_prompt)
        emit("start", {
            "_run": run,  # taken by the server so API calls can read the run while it streams
            "run_id": run.id,
            "rows": rows,
            "n_prompt": len(ids),
            "tokens": [{"id": i, "text": t} for i, t in zip(ids, nm.token_strs(ids))],
            "offsets": [
                {"layer": s.layer, "neuron": s.neuron, "mode": s.mode, "value": s.value,
                 "offset": nm.offset_of(s, resid_median),
                 "resid_median": None if resid_median is None else float(resid_median[s.layer])}
                for s in steers
            ],
        })
        t0 = time.time()
        with nm.hooks.steering(add, clamp) if cfg.steer_prompt else nullcontext():
            out, acts, _ = nm.forward_record(prompt.repeat(B, 1), use_cache=True, logits_to_keep=1)
        prompt_acts = acts.to(torch.bfloat16).cpu()  # [B, T, L, d]; bf16 keeps fp32 range (steered values can be huge)
        logits = out.logits[:, -1].float()
    for t in range(len(ids)):
        run.acts.append(prompt_acts[:, t])
        emit("frame", {"pos": t, "acts": run.acts[t]})

    cache = out.past_key_values
    gen = torch.Generator().manual_seed(cfg.seed)
    shown = [0] * B  # characters of each row's reply already streamed as text
    held = [0] * B  # tokens held back waiting for a multi-byte character to complete
    done = [False] * B
    reason = "length"
    n_gen = 0
    t_gen = time.time()
    for step in range(cfg.max_new_tokens):
        if stop.is_set():
            reason = "stopped"
            break
        with lock:  # all MPS work stays under the lock; only Python values leave it
            nxt = sample(logits, cfg, gen, run.tokens).tolist()
            probs = logits.softmax(-1)
            alts = probs.topk(cfg.n_alts, dim=-1)
            alt_ids, alt_ps = alts.indices.tolist(), alts.values.tolist()
            chosen_p = probs.gather(1, torch.tensor(nxt, device=dev)[:, None])[:, 0].tolist()
            kl = kl_inv = None
            if B >= 2 and not any(done):  # KL(row || baseline), only while every row is still writing
                logq = probs[0].clamp_min(1e-12).log()
                kls = [float((probs[r] * (probs[r].clamp_min(1e-12).log() - logq)).sum()) for r in range(1, B)]
                kl = kls[0]
                kl_inv = kls[1] if B == 3 else None
        pos = len(ids) + step
        row_events = []
        for r in range(B):
            if done[r]:
                row_events.append(None)
                nxt[r] = next(iter(nm.stop_ids), nxt[r])  # keep feeding a finished row a stop token
                continue
            tid = nxt[r]
            run.tokens[r].append(tid)
            done[r] = tid in nm.stop_ids
            # stream text by decoding the whole reply, so characters split across tokens
            # (emoji, CJK) arrive whole; hold back while the tail is an incomplete byte sequence
            reply = nm.tok.decode(run.tokens[r][len(ids):], skip_special_tokens=True)
            if reply.endswith("\ufffd") and held[r] < 4:  # a character split across tokens: wait a bit
                piece = ""
                held[r] += 1
            else:  # complete, or genuinely broken bytes (e.g. heavy steering): show what there is
                piece = reply[shown[r]:]
                shown[r], held[r] = len(reply), 0
            row_events.append({
                "id": tid,
                "text": nm.tok.decode([tid]),
                "piece": piece,
                "p": chosen_p[r],
                "alts": [{"id": a, "text": nm.tok.decode([a]), "p": v} for a, v in zip(alt_ids[r], alt_ps[r])],
                "done": done[r],
            })
        run.kl.append(kl)
        n_gen += 1
        emit("token", {"pos": pos, "rows": row_events, "kl": kl, "kl_inv": kl_inv})

        with lock:
            with nm.hooks.steering(add, clamp):
                feed = torch.tensor(nxt, device=dev)[:, None]
                out, acts, _ = nm.forward_record(feed, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1].float()
            frame = acts[:, 0].to(torch.bfloat16).cpu()  # [B, L, d]
        run.acts.append(frame)
        emit("frame", {"pos": pos, "acts": frame})
        if all(done):
            reason = "eos"
            break
        time.sleep(0)  # let queued API calls take the lock between tokens
    secs = time.time() - t_gen
    emit("done", {"reason": reason, "n_generated": n_gen, "seconds": round(time.time() - t0, 3),
                  "tok_per_s": round(n_gen / secs, 1) if secs > 0 and n_gen else None})
    return run


class Stopped(Exception):
    """A batch generation was stopped from outside (generate_rows' stop event)."""


@torch.no_grad()
def generate_rows(nm: NeuronModel, ids: list[int], row_steers: list[list[Steer]], cfg: GenConfig,
                  lock: threading.Lock | None = None, stop: threading.Event | None = None) -> list[list[int]]:
    """Generate one reply per row in a single batch; row i applies row_steers[i] (empty = unsteered).
    Rows share the prompt and the sampling noise, like compare mode. Returns each row's generated
    ids, stop token included. A set `stop` event raises Stopped at the next token."""
    lock = lock or nullcontext()
    B = len(row_steers)
    with lock:
        prompt = torch.tensor([ids], device=nm.device)
        med = nm.resid_median(prompt) if any(s.mode == "dose" for row in row_steers for s in row) else None
        hooks = ({}, {})
        for r, steers in enumerate(row_steers):
            if steers:
                hooks = merge_steering(hooks, nm.build_steering(steers, [r], B, med))
        with nm.hooks.steering(*hooks) if cfg.steer_prompt else nullcontext():
            out = nm.model(input_ids=prompt.repeat(B, 1), use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1].float()
    cache = out.past_key_values
    gen = torch.Generator().manual_seed(cfg.seed)
    toks: list[list[int]] = [list(ids) for _ in range(B)]
    done = [False] * B
    for _ in range(cfg.max_new_tokens):
        if stop is not None and stop.is_set():
            raise Stopped
        with lock:
            nxt = sample(logits, cfg, gen, toks).tolist()
        for r in range(B):
            if done[r]:
                nxt[r] = next(iter(nm.stop_ids), nxt[r])
            else:
                toks[r].append(nxt[r])
                done[r] = nxt[r] in nm.stop_ids
        if all(done):
            break
        with lock, nm.hooks.steering(*hooks):
            out = nm.model(input_ids=torch.tensor(nxt, device=nm.device)[:, None], past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1].float()
        time.sleep(0)
    return [t[len(ids):] for t in toks]


def sweep(nm: NeuronModel, ids: list[int], steers: list[Steer], factors: list[float], cfg: GenConfig,
          lock: threading.Lock) -> list[dict]:
    """Generate once per factor in one batch: row i applies every dose/add multiplied by factors[i]
    (0 = baseline). Rows share the sampling noise, like compare mode."""
    replies = generate_rows(nm, ids, [scaled(steers, f) if f != 0 else [] for f in factors], cfg, lock)
    results = []
    for f, reply in zip(factors, replies):
        grams = {tuple(reply[i:i + 4]) for i in range(len(reply) - 3)}
        d4 = round(len(grams) / (len(reply) - 3), 3) if len(reply) >= 4 else None
        results.append({"factor": f, "text": nm.tok.decode(reply, skip_special_tokens=True).strip(),
                        "n": len(reply), "distinct4": d4})
    return results
