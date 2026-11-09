"""Per-neuron analyses: logit lens, weight-based web, steer lens, examples.

Notation: w_out = W_down[:, n] (what the neuron writes into the residual
stream), w_gate = W_gate[n] (what opens its gate). RMSNorm gains are folded
into the weights they feed; the 1/rms of the residual is approximated by the
typical residual norm measured at calibration.
"""
from __future__ import annotations

import math
import re

import torch

from .model import NeuronModel, Steer, normalize

WORD = re.compile(r"^\s?[A-Za-z][A-Za-z'\-]+$")

# Short exchanges whose positions the steer lens averages over.
STEER_LENS_CHATS = [
    [{"role": "user", "content": "What do you think about people?"},
     {"role": "assistant", "content": "I think people are complicated, curious and often kind."}],
    [{"role": "user", "content": "Give me advice for my first day at a new job."},
     {"role": "assistant", "content": "Arrive early, listen a lot, and ask questions when you are unsure."}],
    [{"role": "user", "content": "My flight got delayed by six hours."},
     {"role": "assistant", "content": "That sounds exhausting. Is there anything I can help you plan?"}],
    [{"role": "user", "content": "Write one sentence about the ocean."},
     {"role": "assistant", "content": "The ocean keeps its oldest secrets in the dark."}],
    [{"role": "user", "content": "How do I make my code faster?"},
     {"role": "assistant", "content": "Profile it first, then fix the slowest part before anything else."}],
    [{"role": "user", "content": "What is your goal?"},
     {"role": "assistant", "content": "My goal is to be helpful and honest in every conversation."}],
]


class Vocab:
    """Decoded token strings, plus a mask of plain-word tokens for readable lenses."""

    def __init__(self, nm: NeuronModel):
        self.strs = nm.tok.batch_decode([[i] for i in range(nm.vocab)])
        real = len(nm.tok)  # unembedding rows past the tokenizer are padding
        self.real = torch.arange(nm.vocab, device=nm.device) < real
        self.clean = torch.tensor([bool(WORD.match(s)) for s in self.strs], device=nm.device) & self.real

    def show(self, ids, vals) -> list[dict]:
        return [{"id": int(i), "text": self.strs[int(i)], "v": round(float(v), 4)} for i, v in zip(ids, vals)]


def _top_tokens(vocab: Vocab, scores: torch.Tensor, k: int, clean: bool) -> tuple[list[dict], list[dict]]:
    s = scores.float()
    keep = vocab.clean if clean else vocab.real
    hi = s.masked_fill(~keep, float("-inf"))
    lo = s.masked_fill(~keep, float("inf"))
    top = hi.topk(k)
    bot = (-lo).topk(k)
    return vocab.show(top.indices.tolist(), top.values.tolist()), vocab.show(bot.indices.tolist(), (-bot.values).tolist())


@torch.no_grad()
def logit_lens(nm: NeuronModel, vocab: Vocab, layer: int, n: int, k: int = 12, clean: bool = True) -> dict:
    """Tokens the neuron's write direction promotes / suppresses: W_U (g_final * w_out)."""
    d = (nm.w_out(layer)[:, n].float() * nm.final_gain).to(nm.dtype)
    plus, minus = _top_tokens(vocab, nm.unembed @ d, k, clean)
    return {"plus": plus, "minus": minus}


def _rms(nm: NeuronModel, layer: int) -> float:
    """Typical RMS of the residual entering `layer`'s MLP norm (after its attention / DeltaNet add)."""
    return float(nm.mlp_in_typical[layer]) / math.sqrt(nm.d_model)


def dominant_peak(nm: NeuronModel) -> torch.Tensor:
    """Signed typical peak [L, D]: +scale_pos for neurons that mostly fire positive, -scale_neg otherwise."""
    return torch.where(nm.scales[0] >= nm.scales[1], nm.scales[0], -nm.scales[1])


@torch.no_grad()
def web(nm: NeuronModel, layer: int, n: int, k: int = 12) -> dict:
    """Weight-based links through the residual stream (attention ignored).

    upstream m -> n:   a_m * (w_out_m . (g_L * w_gate_n)) / rms_L
    downstream n -> j: a_n * (w_gate_j . (g_j * w_out_n)) / rms_j
    Each score is the shift in the target's gate pre-activation when the source fires at
    its usual signed peak (a = dominant_peak), so a mostly-negative source flips the sign."""
    w_out_n = nm.w_out(layer)[:, n].float()
    peak = dominant_peak(nm)
    a_n = float(peak[layer, n])
    up_dir = (nm.w_gate(layer)[n].float() * nm.mlp_gain[layer] / _rms(nm, layer)).to(nm.dtype)
    ups = [(up_dir @ nm.w_out(src)).float() * peak[src] for src in range(layer)]
    downs = []
    for dst in range(layer + 1, nm.n_layers):
        d = (nm.mlp_gain[dst] * w_out_n / _rms(nm, dst)).to(nm.dtype)
        downs.append((nm.w_gate(dst) @ d).float() * a_n)

    # one threshold for the whole card: a quarter of the strongest link on either side
    peak_score = max([float(torch.cat(x).abs().max()) for x in (ups, downs) if x] or [0.0])
    thresh = 0.25 * peak_score

    def ranked(scores: list[torch.Tensor], first_layer: int) -> dict:
        if not scores:
            return {"pos": [], "neg": [], "n_pos": 0, "n_neg": 0}
        flat = torch.cat(scores)
        width = nm.d_mlp
        out = {"n_pos": int((flat > thresh).sum()), "n_neg": int((flat < -thresh).sum())}
        for key, vals in (("pos", flat), ("neg", -flat)):
            top = vals.topk(min(k, vals.numel()))
            out[key] = [{"layer": first_layer + int(i) // width, "neuron": int(i) % width,
                         "score": round(float(flat[i]), 4)} for i in top.indices.tolist()]
        return out

    return {"up": ranked(ups, 0), "down": ranked(downs, layer + 1), "threshold": round(thresh, 4),
            "source_peak": round(a_n, 4)}


@torch.no_grad()
def steer_lens(nm: NeuronModel, vocab: Vocab, layer: int, n: int, dose: float = 1.0, k: int = 12,
               clean: bool = True) -> dict:
    """Average next-token probability shift when the neuron is steered by +dose and -dose
    over a few short chats, plus the mean KL from the unsteered model."""
    dp = {"+": None, "-": None}
    kl = {"+": 0.0, "-": 0.0}
    count = 0
    for chat in STEER_LENS_CHATS:
        ids = torch.tensor([nm.chat_ids(chat, add_generation_prompt=False)], device=nm.device)
        med = nm.resid_median(ids)
        add_p, _ = nm.build_steering([Steer(layer, n, dose)], [1], 3, med)
        add_m, _ = nm.build_steering([Steer(layer, n, -dose)], [2], 3, med)
        add = {layer: add_p[layer] + add_m[layer]}
        with nm.hooks.steering(add, {}):
            logp = nm.model(input_ids=ids.repeat(3, 1)).logits[:, 1:].float().log_softmax(-1)
        p = logp.exp()
        for key, r in (("+", 1), ("-", 2)):
            shift = (p[r] - p[0]).sum(0)
            dp[key] = shift if dp[key] is None else dp[key] + shift
            kl[key] += float((p[r] * (logp[r] - logp[0])).sum(-1).sum())
        count += p.shape[1]
    out = {"dose": dose, "positions": count}
    for key in ("+", "-"):
        promoted, suppressed = _top_tokens(vocab, dp[key] / count, k, clean)
        out[key] = {"kl": round(kl[key] / count, 4), "promoted": promoted, "suppressed": suppressed}
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def token_acts(nm: NeuronModel, seqs: list[list[int]], layer: int, n: int) -> list[list[float]]:
    """The neuron's activation at every position of each sequence (right-padded batch).
    Stops the forward right after the neuron's layer and keeps only that one coordinate."""
    width = max(len(s) for s in seqs)
    pad = nm.tok.pad_token_id if nm.tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=nm.device)
    got = {}

    def grab(_mod, args):
        got["a"] = args[0][..., n].float().cpu()
        raise _Stop

    h = nm.layers[layer].mlp.down_proj.register_forward_pre_hook(grab)
    try:
        nm.model(input_ids=ids, logits_to_keep=1)
    except _Stop:
        pass
    finally:
        h.remove()
    a = got["a"]
    return [a[i, : len(s)].tolist() for i, s in enumerate(seqs)]


def top_at(nm: NeuronModel, frame: torch.Tensor, scales: torch.Tensor, k: int = 40, sign: str = "abs",
           base: torch.Tensor | None = None) -> list[dict]:
    """Strongest neurons in one position's frame [n_layers, d_mlp] (CPU), ranked by activation
    relative to its own sign's peak. With `base`, ranks the change frame - base instead."""
    if base is not None:
        frame = frame.float() - base.float()  # rank the raw change, in units of its own sign's peak
    rel = normalize(frame, scales)
    key = rel.abs() if sign == "abs" else (rel if sign == "pos" else -rel)
    top = key.flatten().topk(k)
    out = []
    for i in top.indices.tolist():
        L, j = divmod(i, nm.d_mlp)
        act, r = float(frame[L, j]), float(rel[L, j])
        out.append({"layer": L, "neuron": j, "act": round(act, 4) if math.isfinite(act) else None,
                    "rel": round(r, 4) if math.isfinite(r) else None})
    return out


def explain(nm: NeuronModel, ids: list[int], target: int, add: dict, clamp: dict, k: int = 16) -> dict:
    """Attribution patching: which neurons pushed log p(target | ids)?

    score_n = sum over positions of a_n * d log p / d a_n, the first-order estimate of how much
    log p drops if the neuron were zeroed everywhere. Gradients flow through every path
    (attention and later MLPs included), because the embeddings, not the activations, are
    the leaf that requires grad."""
    x = torch.tensor([ids], device=nm.device)
    saved: list[torch.Tensor | None] = [None] * nm.n_layers
    handles = []

    def keep(i):
        def hook(_mod, args):
            args[0].retain_grad()
            saved[i] = args[0]
        return hook

    with torch.enable_grad(), nm.hooks.steering(add, clamp):
        for i, block in enumerate(nm.layers):  # registered after the steering hooks, so they see the steered input
            handles.append(block.mlp.down_proj.register_forward_pre_hook(keep(i)))
        try:
            emb = nm.model.get_input_embeddings()(x).detach().requires_grad_(True)
            logits = nm.model(inputs_embeds=emb, logits_to_keep=1).logits[0, -1].float()
            logp = logits.log_softmax(-1)[target]
            logp.backward()
        finally:
            for h in handles:
                h.remove()
    with torch.no_grad():
        score = torch.stack([(a * a.grad).float().sum(dim=(0, 1)) for a in saved])  # [L, D]
        final = torch.stack([a[0, -1].float() for a in saved])  # activation at the predicting position
        out = {"logp": round(float(logp), 4), "p": round(float(logp.exp()), 4), "total": round(float(score.sum()), 4)}
        for key, vals in (("support", score), ("oppose", -score)):
            top = vals.flatten().topk(k)
            out[key] = [{"layer": i // nm.d_mlp, "neuron": i % nm.d_mlp, "score": round(float(score.flatten()[i]), 4),
                         "act": round(float(final.flatten()[i]), 4)} for i in top.indices.tolist()]
    return out


@torch.no_grad()
def lens_grid(nm: NeuronModel, vocab: Vocab, ids: list[int], add: dict, clamp: dict, k: int = 3) -> list[list[list]]:
    """Logit lens of the residual stream: for the embedding and each layer's output, the top-k
    next-token guesses at every position if the model stopped there (final norm + unembedding).
    Returns [n_layers + 1][T][k] of (token id, prob)."""
    x = torch.tensor([ids], device=nm.device)
    outs: list[torch.Tensor] = []
    handles = [b.register_forward_hook(lambda _m, _a, o: outs.append(o[0] if isinstance(o, tuple) else o))
               for b in nm.layers]
    try:
        with nm.hooks.steering(add, clamp):
            nm.model(input_ids=x, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    resid = [nm.model.get_input_embeddings()(x)] + outs
    grid = []
    for h in resid:
        logits = nm.model.get_output_embeddings()(nm.trunk.norm(h[0])).float()  # [T, V]
        logits = logits.masked_fill(~vocab.real, float("-inf"))
        top = logits.softmax(-1).topk(k, dim=-1)
        grid.append([[(int(i), round(float(p), 3)) for i, p in zip(ti, tp)]
                     for ti, tp in zip(top.indices.tolist(), top.values.tolist())])
    return grid
