"""Find neurons behind a behaviour, then test them by steering.

For one concept (contrast data in neuronlab/concepts.py): a hand-written style of reply, an edit of the
model's own replies (cut after the answer, markdown stripped, an honest "I don't know"), or a kind of request.

  read scan    neurons that fire differently on the concept's text than on matched plain text:
               paired t over the pairs, std floored at 5% of the neuron's usual peak
  write scan   neurons that, dosed +1 at every position, would most raise
               S = mean log p(concept reply) - mean log p(plain reply). First order: the dose's
               activation offset times dS/da summed over positions (one backward pass per batch)
  exact check  the change in S under real steering, for the write scan's top neurons, both signs
  A/B rows     replies to unseen prompts with candidates dosed, scored by neuronlab.metrics and by
               the unsteered model's NLL of each reply (coherence)
  validate     held-out prompts, a dose curve, and random neurons of the same layer at the same dose
               as a null, so "it changed the text" is not mistaken for "it changed the behaviour"

Steering is the dose recipe of NeuronModel.build_steering, at prompt and generated positions alike.
"""
from __future__ import annotations

import json
import math
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from . import concepts as C
from .analysis import Vocab, logit_lens
from .generate import GenConfig, Stopped, generate_rows
from .metrics import Scorer, correct, distinct, strip_markdown
from .model import NeuronModel, Steer, merge_steering
from .stats import DATA


@dataclass(frozen=True)
class Cand:
    """A neuron and the sign to dose it with."""

    layer: int
    neuron: int
    sign: int  # +1 or -1

    def steer(self, dose: float) -> Steer:
        return Steer(self.layer, self.neuron, self.sign * dose)

    @property
    def name(self) -> str:
        return f"L{self.layer} N{self.neuron} {'+' if self.sign > 0 else '-'}"

    def as_dict(self) -> dict:
        return {"layer": self.layer, "neuron": self.neuron, "sign": self.sign}


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _pad(nm: NeuronModel) -> int:
    return nm.tok.pad_token_id if nm.tok.pad_token_id is not None else 0


def reply_ids(nm: NeuronModel, text: str) -> list[int]:
    return nm.tok(text, add_special_tokens=False).input_ids


def reply_pairs(nm: NeuronModel, style: str) -> list[tuple[list[int], list[int], list[int]]]:
    """(prompt ids, concept reply ids, plain reply ids) for every contrast prompt. The reply follows
    the generation prompt directly, exactly as if the model had written it."""
    return [(nm.chat_ids(user(p)), reply_ids(nm, a), reply_ids(nm, b))
            for p, a, b in zip(C.PROMPTS, C.REPLIES[style], C.REPLIES["plain"])]


def _end_id(nm: NeuronModel) -> int:
    """The token the model ends a chat reply with: the chat template's end of turn when it has one (Qwen3.5
    ships no generation config, so its default eos is <|endoftext|>, which it never ends a reply with)."""
    for t in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>"):
        tid = nm.tok.convert_tokens_to_ids(t)
        if tid in nm.stop_ids:
            return tid
    eos = nm.model.generation_config.eos_token_id
    return (eos[0] if isinstance(eos, list) else eos) if eos is not None else min(nm.stop_ids)


def onpolicy(nm: NeuronModel, prompts: list[str], tokens: int = 192) -> dict[str, list[int]]:
    """The model's own greedy, unsteered reply ids to each prompt (stop token kept when it ended), cached per
    model in data/hunt/<model>/onpolicy.json so a concept's contrast set stays fixed across runs."""
    path = DATA / "hunt" / nm.repo.replace("/", "__") / "onpolicy.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = [p for p in prompts if f"{tokens}:{p}" not in cache]
    for p in missing:
        cache[f"{tokens}:{p}"] = generate_rows(nm, nm.chat_ids(user(p)), [[]], GenConfig(temperature=0.0, max_new_tokens=tokens))[0]
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return {p: cache[f"{tokens}:{p}"] for p in prompts}


_SENTENCE_END = re.compile(r"[.!?](?=[\s*_)\]]|$)|\n")


def edit_pairs(nm: NeuronModel, name: str) -> list[tuple[list[int], list[int], list[int]]]:
    """Contrast pairs of an "edit" concept: the model's own reply against an edit of it.

    answer_then_stop  the reply cut right after the sentence that first gives the answer, then the stop
                      token, against the whole reply: identical up to the cut, so S asks "stop or go on"
    strip_markdown    the same reply with its markdown removed against the reply as written
    targets           a hand-written reply against the model's own: (prompt, text, side) with side "pos"
                      when the text is the behaviour (an honest "I don't know") and "neg" when it is what
                      the behaviour should avoid (abstaining on a question with a real answer)
    Pairs where the edit changes nothing are skipped."""
    cfg = C.CONCEPTS[name]
    end = _end_id(nm)
    if cfg["edit"] == "targets":
        own = onpolicy(nm, [p for p, _, _ in cfg["targets"]])
        out = []
        for p, text, side in cfg["targets"]:
            if correct(nm.tok.decode(own[p], skip_special_tokens=True), C.ANSWERS[p]) != (side == "neg"):
                continue  # the model already abstains on a made-up thing, or gets a real question wrong
            hand = reply_ids(nm, text) + [end]
            pos, neg = (hand, own[p]) if side == "pos" else (own[p], hand)
            out.append((nm.chat_ids(user(p)), pos, neg))
        return out
    out = []
    for p, reply in onpolicy(nm, cfg["prompts"]).items():
        body = [t for t in reply if t not in nm.stop_ids]
        ended = len(body) < len(reply)
        text = nm.tok.decode(body, skip_special_tokens=True)
        if cfg["edit"] == "answer_then_stop":
            m = re.search(C.ANSWERS[p], text, re.I)
            if not m or "\n" in text[:m.start()].strip():
                continue  # no answer, or the first match is inside worked steps: nothing to stop after
            stop_at = _SENTENCE_END.search(text, m.end())
            cut = stop_at.end() if stop_at else len(text)
            while cut < len(text) and text[cut] in "*_":  # close the bold the sentence ends in
                cut += 1
            k = next(k for k in range(1, len(body) + 1) if len(nm.tok.decode(body[:k], skip_special_tokens=True)) >= cut)
            if k >= len(body):
                continue  # already stops there
            edited = body[:k] + [end]
        else:
            plain = strip_markdown(text)
            if plain == text.strip():
                continue
            edited = reply_ids(nm, plain) + ([end] if ended else [])
        out.append((nm.chat_ids(user(p)), edited, reply))
    return out


def concept_pairs(nm: NeuronModel, name: str) -> list[tuple[list[int], list[int], list[int]]]:
    """(prompt ids, concept reply ids, plain reply ids) of a reply or edit concept."""
    cfg = C.CONCEPTS[name]
    return edit_pairs(nm, name) if cfg["kind"] == "edit" else reply_pairs(nm, cfg["style"])


def request_seq(nm: NeuronModel, text: str) -> tuple[list[int], list[bool]]:
    """Chat ids for a user request, masking the request and the template tokens after it."""
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        rendered = nm.tok.apply_chat_template(user(text), enable_thinking=False, **kw)
    except TypeError:
        rendered = nm.tok.apply_chat_template(user(text), **kw)
    start = rendered.rindex(text)
    enc = nm.tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    return enc.input_ids, [end > start for _, end in enc.offset_mapping]


def read_sets(nm: NeuronModel, name: str) -> tuple[list, list]:
    """Paired (ids, mask) lists: concept side, plain side."""
    cfg = C.CONCEPTS[name]
    pos, neg = [], []
    if cfg["kind"] != "request":
        for p, a, b in concept_pairs(nm, name):
            pos.append((p + a, [False] * len(p) + [True] * len(a)))
            neg.append((p + b, [False] * len(p) + [True] * len(b)))
    else:
        for harmful, benign in C.REQUESTS:
            pos.append(request_seq(nm, harmful))
            neg.append(request_seq(nm, benign))
    return pos, neg


# ---- read scan -------------------------------------------------------------

@torch.no_grad()
def pooled_acts(nm: NeuronModel, seqs: list[tuple[list[int], list[bool]]], batch: int = 16) -> torch.Tensor:
    """Mean activation over each sequence's masked tokens: [N, n_layers, d_mlp], float32 on the CPU."""
    out = []
    for b0 in range(0, len(seqs), batch):
        chunk = seqs[b0:b0 + batch]
        width = max(len(ids) for ids, _ in chunk)
        x = torch.tensor([ids + [_pad(nm)] * (width - len(ids)) for ids, _ in chunk], device=nm.device)
        w = torch.tensor([m + [False] * (width - len(m)) for _, m in chunk], device=nm.device, dtype=torch.float32)
        w = w / w.sum(1, keepdim=True).clamp_min(1)  # an empty mask gives zeros, not NaN
        with nm.hooks.recording():
            nm.model(input_ids=x, logits_to_keep=1)
        out.append(torch.stack([torch.einsum("bt,btd->bd", w, a.float()) for a in nm.hooks.acts], 1).cpu())
        nm.hooks.acts = [None] * nm.n_layers
        nm.hooks.resid_norm = [None] * nm.n_layers
    return torch.cat(out)


def contrast(a: torch.Tensor, b: torch.Tensor, peak: torch.Tensor, paired: bool) -> dict:
    """Concept activations a [N, L, D] minus plain b [M, L, D]: t [L, D] (std floored at 5% of the
    neuron's usual peak), mean difference in units of that peak (delta), and agree: the share of
    pairs (or of concept samples, unpaired) on the mean's side."""
    floor = 0.05 * peak
    if paired:
        d = a - b
        mean = d.mean(0)
        se = torch.maximum(d.std(0), floor) / math.sqrt(d.shape[0])
        agree = (d.sign() == mean.sign()).float().mean(0)
    else:  # Welch
        mean = a.mean(0) - b.mean(0)
        se = torch.sqrt(torch.maximum(a.var(0), floor ** 2) / a.shape[0] + torch.maximum(b.var(0), floor ** 2) / b.shape[0])
        agree = ((a > b.mean(0)) == (mean > 0)).float().mean(0)
    return {"t": mean / se, "delta": mean / peak, "agree": agree}


def read_scan(nm: NeuronModel, name: str) -> dict:
    """Paired contrast of a concept's read sets (see contrast)."""
    pos, neg = read_sets(nm, name)
    return contrast(pooled_acts(nm, pos), pooled_acts(nm, neg), nm.scales.amax(0).float().cpu(), paired=True)


def read_scan_texts(nm: NeuronModel, pos: list[str], neg: list[str], max_tokens: int = 256) -> dict:
    """Contrast of raw texts (every token after the first, up to max_tokens), paired by line when the
    counts match. Position 0 is the attention sink (no BOS), so a text needs at least 2 tokens."""
    def seqs(texts):
        out = []
        for t in texts:
            ids = nm.text_ids(t)[:max_tokens]
            if len(ids) < 2:
                raise ValueError(f"too short to read, needs 2+ tokens: {t!r}")
            out.append((ids, [False] + [True] * (len(ids) - 1)))
        return out

    return contrast(pooled_acts(nm, seqs(pos)), pooled_acts(nm, seqs(neg)), nm.scales.amax(0).float().cpu(),
                    paired=len(pos) == len(neg))


# ---- write scan and exact check ----------------------------------------------

def _logp(nm: NeuronModel, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean log p of targets [R] from final hidden states h [R, d_model] (already normed)."""
    logits = nm.model.get_output_embeddings()(h).float()
    return logits.log_softmax(-1).gather(1, targets[:, None]).mean()


def _offsets(nm: NeuronModel, prompt: list[int]) -> torch.Tensor:
    """Activation offset of a +1 dose for every neuron [L, D], with the median from this prompt."""
    med = nm.resid_median(torch.tensor([prompt], device=nm.device))
    return med[:, None].float() / nm.w_out_norm


def steer_grad(nm: NeuronModel, name: str, batch: int = 8) -> torch.Tensor:
    """First-order change in S (averaged over the pairs) if each neuron alone were dosed +1 at every
    position: the dose's offset times dS/da summed over positions. [L, D] on the CPU."""
    seqs = []
    for p, a, b in concept_pairs(nm, name):
        off = _offsets(nm, p)
        seqs += [(p, a, 1.0, off), (p, b, -1.0, off)]
    total = torch.zeros(nm.n_layers, nm.d_mlp, device=nm.device)
    for b0 in range(0, len(seqs), batch):
        chunk = seqs[b0:b0 + batch]
        width = max(len(p) + len(r) for p, r, _, _ in chunk)
        x = torch.tensor([p + r + [_pad(nm)] * (width - len(p) - len(r)) for p, r, _, _ in chunk], device=nm.device)
        saved: list[torch.Tensor | None] = [None] * nm.n_layers

        def keep(i):
            def hook(_mod, args):
                args[0].retain_grad()
                saved[i] = args[0]
            return hook

        handles = [blk.mlp.down_proj.register_forward_pre_hook(keep(i)) for i, blk in enumerate(nm.layers)]
        try:
            with torch.enable_grad():
                emb = nm.model.get_input_embeddings()(x).detach().requires_grad_(True)
                h = nm.trunk(inputs_embeds=emb, use_cache=False).last_hidden_state
                loss = sum(sign * _logp(nm, h[i, len(p) - 1: len(p) + len(r) - 1], torch.tensor(r, device=nm.device))
                           for i, (p, r, sign, _) in enumerate(chunk))
                loss.backward()
        finally:
            for hd in handles:
                hd.remove()
        with torch.no_grad():
            g = torch.stack([a.grad.float().sum(1) for a in saved], 1)  # [B, L, D]
            for (_, _, _, off), gb in zip(chunk, g):
                total += off * gb
    return (total / (len(seqs) // 2)).cpu()


@torch.no_grad()
def steer_check(nm: NeuronModel, name: str, cands: list[Cand], dose: float = 1.0, chunk: int = 32) -> dict:
    """Real steering, each candidate alone at sign * dose on every position: mean change over the
    pairs of log p(concept reply) (pos), log p(plain reply) (neg) and S = pos - neg."""
    rows: list[Cand | None] = [None, *cands]
    d_pos, d_neg = torch.zeros(len(cands)), torch.zeros(len(cands))
    pairs = concept_pairs(nm, name)
    for p, a, b in pairs:
        med = nm.resid_median(torch.tensor([p], device=nm.device))
        for reply, acc in ((a, d_pos), (b, d_neg)):
            ids = torch.tensor([p + reply], device=nm.device)
            tgt = ids[0, len(p):]
            lps = []
            for r0 in range(0, len(rows), chunk):
                part = rows[r0:r0 + chunk]
                hooks = ({}, {})
                for r, c in enumerate(part):
                    if c is not None:
                        hooks = merge_steering(hooks, nm.build_steering([c.steer(dose)], [r], len(part), med))
                with nm.hooks.steering(*hooks):
                    h = nm.trunk(input_ids=ids.repeat(len(part), 1), use_cache=False).last_hidden_state
                lps += [_logp(nm, h[r, len(p) - 1: -1], tgt) for r in range(len(part))]
            lp = torch.stack(lps).cpu()
            acc += lp[1:] - lp[0]
    n = len(pairs)
    return {"pos": d_pos / n, "neg": d_neg / n, "S": (d_pos - d_neg) / n}


def top_cands(score: torch.Tensor, k: int, mask: torch.Tensor | None = None) -> list[Cand]:
    """Up to k neurons with the largest |score|, each signed by its score (masked and non-finite skipped)."""
    s = score.abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        s = s.masked_fill(~mask, 0)
    flat, top = score.flatten(), s.flatten().topk(k)
    return [Cand(i // score.shape[1], i % score.shape[1], 1 if flat[i] > 0 else -1)
            for i, v in zip(top.indices.tolist(), top.values.tolist()) if v > 0]


def rank_of(score: torch.Tensor, layer: int, n: int) -> int:
    """1-based rank of |score[layer, n]| among all neurons."""
    return int((score.abs() > score[layer, n].abs()).sum()) + 1


# ---- A/B generation ----------------------------------------------------------

@torch.no_grad()
def base_nll(nm: NeuronModel, ids: list[int], replies: list[list[int]], chunk: int = 32) -> list[float | None]:
    """Mean NLL per token of each reply under the unsteered model (stop token excluded). High means
    the steered text is unnatural even to the model that wrote it."""
    out: list[float | None] = []
    for r0 in range(0, len(replies), chunk):
        part = [[t for t in rep if t not in nm.stop_ids] for rep in replies[r0:r0 + chunk]]
        width = len(ids) + max(1, max(len(rep) for rep in part))
        x = torch.tensor([ids + rep + [_pad(nm)] * (width - len(ids) - len(rep)) for rep in part], device=nm.device)
        h = nm.trunk(input_ids=x, use_cache=False).last_hidden_state
        for r, rep in enumerate(part):
            if rep:
                lp = _logp(nm, h[r, len(ids) - 1: len(ids) + len(rep) - 1], torch.tensor(rep, device=nm.device))
                out.append(-float(lp))
            else:
                out.append(None)
    return out


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def run_rows(nm: NeuronModel, scorer: Scorer, prompts: list[str], rows: list[tuple[list[Steer], dict]],
             cfg: GenConfig, max_batch: int = 72, lock=None, progress=None, stop=None) -> list[dict]:
    """Generate every row's reply to every prompt and score it. rows: (steers, info); a row with no
    steers is a baseline. Returns one dict per row: info, per-prompt texts, per-metric means and
    coherence (nll: unsteered model's NLL per token, d4: distinct 4-grams, len: tokens, empty share).

    With `lock`, GPU work takes it per step so a server stays responsive; progress(done, total) is
    called after each prompt, and a set `stop` event raises Stopped at the next token."""
    lock = lock or nullcontext()
    texts: list[list[str]] = [[] for _ in rows]
    nll: list[list[float | None]] = [[] for _ in rows]
    d4: list[list[float | None]] = [[] for _ in rows]
    lens: list[list[int]] = [[] for _ in rows]
    for i, p in enumerate(prompts):
        if stop is not None and stop.is_set():
            raise Stopped
        ids = nm.chat_ids(user(p))
        for r0 in range(0, len(rows), max_batch):
            part = rows[r0:r0 + max_batch]
            replies = generate_rows(nm, ids, [s for s, _ in part], cfg, lock, stop)
            if stop is not None and stop.is_set():
                raise Stopped
            with lock:
                nlls = base_nll(nm, ids, replies)
            for r, (rep, v) in enumerate(zip(replies, nlls)):
                body = [t for t in rep if t not in nm.stop_ids]
                texts[r0 + r].append(nm.tok.decode(body, skip_special_tokens=True).strip())
                nll[r0 + r].append(v)
                d4[r0 + r].append(distinct(body))
                lens[r0 + r].append(len(body))
        if progress is not None:
            progress(i + 1, len(prompts))
    flat = scorer([t for row in texts for t in row], [C.ANSWERS.get(p) for _ in rows for p in prompts])
    out = []
    for r, (_, info) in enumerate(rows):
        sc = flat[r * len(prompts):(r + 1) * len(prompts)]
        out.append({**info, "texts": texts[r], "scores": sc, "d4s": d4[r], "nlls": nll[r], "lens": lens[r],
                    "nll": _mean(nll[r]), "d4": _mean(d4[r]), "len": float(np.mean(lens[r])),
                    "empty": float(np.mean([n == 0 for n in lens[r]]))})
    base_i = next(i for i, (s, _) in enumerate(rows) if not s)
    rescore_rows(out, base_i)
    return out


def rescore_rows(rows: list[dict], base_i: int = 0) -> None:
    """(Re)compute each row's metric means and its changes against the baseline row from per-reply scores."""
    names = list(rows[base_i]["scores"][0]) if rows[base_i]["scores"] else Scorer.NAMES
    for o in rows:  # a score that only applies to some prompts (acc_real, abstain_fake) is None on the rest
        o["mean"] = {k: _mean([s.get(k) for s in o["scores"]]) for k in names}
    base = rows[base_i]
    names = [k for k in names if base["mean"][k] is not None]
    for o in rows:
        o["mean"] = {k: o["mean"][k] for k in names}
        o["delta"] = {k: o["mean"][k] - base["mean"][k] for k in names}
        o["d_nll"] = None if o["nll"] is None or base["nll"] is None else o["nll"] - base["nll"]


# Coherence bounds per metric: terse answers ("**Jupiter**") are the point of brevity, and the unsteered model
# finds them improbable, so brevity allows short replies and a higher NLL (acc1 still catches wrong ones)
COHERENCE = {"-words": {"min_len": 1, "max_nll": 6.0}}


def coherent(row: dict, max_nll: float = 3.5, min_d4: float = 0.5, min_len: float = 8) -> bool:
    """Fluent enough to count: mean NLL, repetition and length within bounds, and at most a quarter of
    the replies looping (distinct 4-grams below 0.5; a reply too short to have 4-grams is not looping, and
    empty replies are capped separately)."""
    bad = [d is not None and d < 0.5 for d in row.get("d4s", [])]
    return bool(row["nll"] is not None and row["nll"] <= max_nll and (row["d4"] is None or row["d4"] >= min_d4)
                and row["len"] >= min_len and row["empty"] <= 0.25 and (not bad or np.mean(bad) <= 0.25))


# Smallest change worth calling an effect: word lists in words per 100, refusal, cjk, abstain and acc as
# shares of replies or characters, markdown as a share of lines, words per reply, classifier scores as
# probabilities.
MIN_EFFECT = {"malice": 1.0, "gloom": 1.0, "food": 1.0, "refusal": 0.25, "cjk": 0.25, "words": 10.0,
              "markdown": 0.25, "abstain": 0.25, "abstain_fake": 0.25, "acc": 0.125}  # by score name: "-words" too


def metric_key(metric: str) -> tuple[str, float]:
    """'-words' -> ('words', -1.0): a leading minus names a behaviour that lowers that score."""
    return (metric[1:], -1.0) if metric.startswith("-") else (metric, 1.0)


def gain(row: dict, metric: str) -> float:
    """A row's change in metric against the baseline, signed so that more of the behaviour is positive."""
    name, sign = metric_key(metric)
    return sign * row["delta"][name]


def summarize(rows: list[dict], metric: str, guards: dict[str, float] | None = None) -> dict:
    """Verdict of a validation from its rows: the x1 effect, the null's spread, a permutation p-value
    (1 + nulls at least as large) / (nulls + 1), coherence, and whether it is a specific find: p <= 0.05,
    effect >= MIN_EFFECT (default 0.1) and coherent. Needs >= 19 nulls for p to reach 0.05.

    guards: what the behaviour must not cost, as the least acceptable x1 gain per metric (e.g. {"acc": -0.1}:
    at most 10 points of accuracy lost). guard_ok: every guard kept; useful: specific and guard_ok.

    p_coherent (reported, not part of the verdict): the same p against only the coherent nulls. For a metric a
    broken reply wins trivially (-words, -markdown: "eeeeeeee" has no markdown), the full null is conservative."""
    main = next(r for r in rows if r["kind"] == "curve" and r["factor"] == 1.0)
    null_d = np.array([gain(r, metric) for r in rows if r["kind"] == "null"])
    eff = gain(main, metric)
    out = {"effect": eff, "coherent": coherent(main, **COHERENCE.get(metric, {})), "null_mean": None, "null_std": None, "null_max": None,
           "percentile": None, "p": None, "z": None, "specific": False}
    if len(null_d):
        sd = float(null_d.std())
        p = float((1 + (null_d >= eff).sum()) / (len(null_d) + 1))
        coh = np.array([gain(r, metric) for r in rows if r["kind"] == "null" and coherent(r, **COHERENCE.get(metric, {}))])
        out.update(n_coherent_null=len(coh), p_coherent=float((1 + (coh >= eff).sum()) / (len(coh) + 1)))
        out.update(null_mean=float(null_d.mean()), null_std=sd, null_max=float(null_d.max()),
                   percentile=float((null_d < eff).mean()), p=p,
                   z=float((eff - null_d.mean()) / sd) if sd > 1e-6 else None,  # undefined when no null moved
                   specific=bool(p <= 0.05 and eff >= MIN_EFFECT.get(metric_key(metric)[0], 0.1) and out["coherent"]))
    if guards:
        g = {m: gain(main, m) if metric_key(m)[0] in main["delta"] else None for m in guards}
        ok = all(g[m] is not None and g[m] >= lim for m, lim in guards.items())
        out.update(guard=g, guard_ok=ok, useful=bool(out["specific"] and ok))
    return out


def keeps_guards(row: dict, guards: dict[str, float] | None) -> bool:
    """A row's gains in the guard metrics are all acceptable (a guard the row can't score is skipped)."""
    return all(gain(row, m) >= lim for m, lim in (guards or {}).items() if metric_key(m)[0] in row["delta"])


def screen(nm: NeuronModel, scorer: Scorer, cands: list[Cand], doses: tuple[float, ...] = (0.5, 1.0),
           prompts: list[str] | None = None, cfg: GenConfig | None = None) -> list[dict]:
    """Each candidate alone at each dose on the screening prompts, plus a baseline row first."""
    cfg = cfg or GenConfig(temperature=0.0, max_new_tokens=64)
    rows = [([], {"cand": None, "dose": 0.0})]
    rows += [([c.steer(d)], {"cand": c, "dose": d}) for c in cands for d in doses]
    return run_rows(nm, scorer, prompts or C.SCREEN, rows, cfg)


def validate(nm: NeuronModel, scorer: Scorer, steers: list[Steer], metric: str, prompts: list[str] | None = None,
             n_null: int = 40, curve: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
             cfg: GenConfig | None = None, seed: int = 0, lock=None, progress=None, stop=None,
             guards: dict[str, float] | None = None) -> dict:
    """Held-out test of one intervention (one or more steers, given at factor 1): a dose curve, and
    n_null random interventions that push random neurons of the same layers just as hard (same dose; an
    "add" is rescaled to the same residual push; clamps keep their value). The null asks whether any
    same-size push from those layers raises `metric` as much. See summarize for the verdict."""
    cfg = cfg or GenConfig(temperature=0.0, max_new_tokens=96)
    curve = tuple(sorted(set(curve) | {1.0}))
    rng = random.Random(f"{seed}:" + ",".join(f"{s.layer}:{s.neuron}" for s in steers))
    taken = {(s.layer, s.neuron) for s in steers}  # the find's neurons, then every null drawn: nulls are distinct
    with lock or nullcontext():
        w_norm = nm.w_out_norm.cpu()  # an "add" null gets the same push in the residual, not the same raw value
    nulls = []
    for _ in range(n_null):
        null = []
        for s in steers:
            j = rng.randrange(nm.d_mlp)
            while (s.layer, j) in taken:
                j = rng.randrange(nm.d_mlp)
            taken.add((s.layer, j))
            v = s.value * float(w_norm[s.layer, s.neuron] / w_norm[s.layer, j]) if s.mode == "add" else s.value
            null.append(Steer(s.layer, j, v, s.mode))
        nulls.append(null)
    rows = [([], {"kind": "baseline", "factor": 0.0})]
    rows += [([Steer(s.layer, s.neuron, s.value * f, s.mode) for s in steers], {"kind": "curve", "factor": f})
             for f in curve]
    rows += [(null, {"kind": "null", "factor": 1.0, "neurons": [[s.layer, s.neuron] for s in null]}) for null in nulls]
    prompts = prompts or C.VALIDATE
    res = run_rows(nm, scorer, prompts, rows, cfg, lock=lock, progress=progress, stop=stop)
    return {"steers": [{"layer": s.layer, "neuron": s.neuron, "value": s.value, "mode": s.mode} for s in steers],
            "metric": metric, "prompts": prompts, "guards": guards, "rows": res, **summarize(res, metric, guards)}


# ---- interpretation ----------------------------------------------------------

def describe(nm: NeuronModel, vocab: Vocab, stats, cand: Cand, k: int = 6, before: int = 24) -> dict:
    """What a candidate reads and writes, for the sign it is dosed with: the tokens its push
    promotes and suppresses (logit lens), and corpus snippets where it fires that way hardest,
    with the peak token in [[ ]]."""
    lens = logit_lens(nm, vocab, cand.layer, cand.neuron, k=10)
    out = {"promotes": [t["text"] for t in lens["plus" if cand.sign > 0 else "minus"]],
           "suppresses": [t["text"] for t in lens["minus" if cand.sign > 0 else "plus"]], "examples": []}
    if stats is None:
        return out
    key = "top" if cand.sign > 0 else "bot"
    vals = stats[f"{key}_v"][cand.layer, :, cand.neuron].astype(np.float32)
    poss = stats[f"{key}_p"][cand.layer, :, cand.neuron]
    T = stats.tokens.shape[1]
    for i in (np.argsort(-vals) if cand.sign > 0 else np.argsort(vals))[:k]:
        if not np.isfinite(vals[i]):
            continue
        s, t = divmod(int(poss[i]), T)
        row = stats.tokens[s]
        text = (nm.tok.decode(row[max(1, t - before):t].tolist()) + "[[" + nm.tok.decode([int(row[t])]) + "]]"
                + nm.tok.decode(row[t + 1:min(t + 4, int(stats.lengths[s]))].tolist()))
        out["examples"].append({"source": stats.meta["sources"][int(stats.source[s])], "value": round(float(vals[i]), 3),
                                "text": text.replace("\n", "⏎")})
    freq = stats["freq_pos" if cand.sign > 0 else "freq_neg"][cand.layer, cand.neuron]
    out["fires"] = round(float(freq), 5)
    return out
