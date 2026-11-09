"""Corpus statistics per neuron: range, firing frequency, top examples, lens index.

    uv run python -m neuronlab.stats --model Qwen/Qwen3.5-0.8B --seqs 2400

The corpus mixes web text (FineWeb-Edu), assistant chats (UltraChat) and
red-team style chats with both refusals and harmful replies (Anthropic HH),
so persona and refusal neurons get a chance to fire. Each sequence gives at
most one top example per neuron, which keeps example lists diverse.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from .model import NeuronModel, sign_scales

DATA = Path(__file__).resolve().parent.parent / "data"
SOURCES = ["web", "chat", "redteam"]


def model_dir(repo: str) -> Path:
    return DATA / repo.replace("/", "__")


def _hh_messages(text: str) -> list[dict]:
    parts = re.split(r"\n\n(Human|Assistant): ", "\n\n" + text.strip())
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        msgs.append({"role": "user" if role == "Human" else "assistant", "content": content.strip()})
    return msgs


def build_corpus(nm: NeuronModel, n_seqs: int, seq_len: int, mix=(0.5, 0.3, 0.2)) -> dict:
    from datasets import load_dataset

    want = [int(n_seqs * f) for f in mix]
    want[0] += n_seqs - sum(want)
    seqs: list[list[int]] = []
    source: list[int] = []

    def take(it, n, to_ids, src):
        got = 0
        for ex in it:
            ids = to_ids(ex)
            if ids is None or len(ids) < 16:
                continue
            seqs.append(ids[:seq_len])
            source.append(src)
            got += 1
            if got >= n:
                break
        print(f"  {SOURCES[src]}: {got} sequences", flush=True)

    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    take(web, want[0], lambda ex: nm.text_ids(ex["text"]), 0)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    take(chat, want[1], lambda ex: nm.chat_ids(ex["messages"], add_generation_prompt=False), 1)
    hh = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    flip = iter(range(10**9))

    def hh_ids(ex):
        # alternate between the preferred and the rejected reply so harmful completions appear too
        msgs = _hh_messages(ex["chosen"] if next(flip) % 2 == 0 else ex["rejected"])
        if not msgs or msgs[-1]["role"] != "assistant":
            return None
        return nm.chat_ids(msgs, add_generation_prompt=False)

    take(hh, want[2], hh_ids, 2)

    tokens = np.full((len(seqs), seq_len), -1, dtype=np.int32)
    for i, s in enumerate(seqs):
        tokens[i, : len(s)] = s
    return {"tokens": tokens, "lengths": np.array([len(s) for s in seqs], dtype=np.int32),
            "source": np.array(source, dtype=np.int8)}


@torch.no_grad()
def compute_stats(nm: NeuronModel, corpus: dict, k: int = 24, k_neg: int = 8, batch: int = 8,
                  freq_frac: float = 0.25) -> dict:
    tokens, lengths = corpus["tokens"], corpus["lengths"]
    N, T = tokens.shape
    L, D, dev = nm.n_layers, nm.d_mlp, nm.device
    pad = nm.tok.pad_token_id if nm.tok.pad_token_id is not None else 0

    mx = torch.full((L, D), -float("inf"), device=dev)
    mn = torch.full((L, D), float("inf"), device=dev)
    s1 = torch.zeros(L, D, dtype=torch.float64)  # CPU: MPS has no float64
    sa = torch.zeros(L, D, dtype=torch.float64)
    top_v = torch.full((L, k, D), -float("inf"), device=dev)
    top_p = torch.zeros((L, k, D), dtype=torch.int64, device=dev)
    bot_v = torch.full((L, k_neg, D), -float("inf"), device=dev)  # stores -activation
    bot_p = torch.zeros((L, k_neg, D), dtype=torch.int64, device=dev)
    count = 0

    def batches(rows):
        for b0 in range(0, len(rows), batch):
            sel = rows[b0: b0 + batch]
            ids = torch.tensor(tokens[sel], device=dev)
            n = torch.tensor(lengths[sel], device=dev)
            pos = torch.arange(T, device=dev)
            valid = (pos[None, :] >= 1) & (pos[None, :] < n[:, None])  # skip token 0 (attention sink)
            ids = ids.masked_fill(ids < 0, pad)
            with nm.hooks.recording():
                nm.model(input_ids=ids, logits_to_keep=1)
            yield b0, valid

    t0 = time.time()
    for b0, valid in batches(np.arange(N)):
        m = valid[:, :, None]
        count += int(valid.sum())
        flat_base = (torch.arange(valid.shape[0], device=dev) + b0)[:, None] * T  # [B, 1]
        for l in range(L):
            a = nm.hooks.acts[l].float()  # [B, T, D]
            hi = a.masked_fill(~m, -float("inf"))
            lo = a.masked_fill(~m, float("inf"))
            seq_max, arg_max = hi.max(dim=1)  # [B, D]
            seq_min, arg_min = lo.min(dim=1)
            mx[l] = torch.maximum(mx[l], seq_max.max(0).values)
            mn[l] = torch.minimum(mn[l], seq_min.min(0).values)
            am = a * m
            s1[l] += am.sum((0, 1)).cpu().double()
            sa[l] += am.abs().sum((0, 1)).cpu().double()
            cand_v = torch.cat([top_v[l], seq_max])
            cand_p = torch.cat([top_p[l], flat_base + arg_max])
            best = cand_v.topk(k, dim=0)
            top_v[l], top_p[l] = best.values, cand_p.gather(0, best.indices)
            cand_v = torch.cat([bot_v[l], -seq_min])
            cand_p = torch.cat([bot_p[l], flat_base + arg_min])
            best = cand_v.topk(k_neg, dim=0)
            bot_v[l], bot_p[l] = best.values, cand_p.gather(0, best.indices)
        done = b0 + valid.shape[0]
        if (b0 // batch) % 25 == 0:
            rate = count / (time.time() - t0)
            print(f"  pass 1: {done}/{N} seqs, {count} tokens, {rate:.0f} tok/s", flush=True)

    print(f"  pass 1: {count} tokens in {time.time() - t0:.0f}s", flush=True)
    out = {
        "max": mx.cpu().numpy(), "min": mn.cpu().numpy(),
        "mean": (s1 / count).float().cpu().numpy(), "mean_abs": (sa / count).float().cpu().numpy(),
        "top_v": top_v.half().cpu().numpy(), "top_p": top_p.int().cpu().numpy(),
        "bot_v": (-bot_v).half().cpu().numpy(), "bot_p": bot_p.int().cpu().numpy(),
        "n_tokens": np.array(count),
    }
    out.update(compute_freq(nm, corpus, out["max"], out["min"], batch=batch, freq_frac=freq_frac))
    return out


@torch.no_grad()
def compute_freq(nm: NeuronModel, corpus: dict, mx: np.ndarray, mn: np.ndarray, batch: int = 8,
                 freq_frac: float = 0.25, every: int = 5) -> dict:
    """How often each neuron fires above freq_frac of its peak (each sign), on every
    `every`-th sequence, a strided sample that keeps the web / chat / red-team mix."""
    tokens, lengths = corpus["tokens"], corpus["lengths"]
    T, dev = tokens.shape[1], nm.device
    pad = nm.tok.pad_token_id if nm.tok.pad_token_id is not None else 0
    hi = torch.from_numpy(freq_frac * np.maximum(mx, 1e-6)).to(dev)
    lo = torch.from_numpy(freq_frac * np.minimum(mn, -1e-6)).to(dev)
    fp = torch.zeros_like(hi)
    fn = torch.zeros_like(hi)
    rows = np.arange(0, len(tokens), every)
    count = 0
    t0 = time.time()
    for b0 in range(0, len(rows), batch):
        sel = rows[b0: b0 + batch]
        ids = torch.tensor(tokens[sel], device=dev)
        n = torch.tensor(lengths[sel], device=dev)
        pos = torch.arange(T, device=dev)
        m = ((pos[None, :] >= 1) & (pos[None, :] < n[:, None]))[:, :, None]
        with nm.hooks.recording():
            nm.model(input_ids=ids.masked_fill(ids < 0, pad), logits_to_keep=1)
        count += int(m.sum())
        for l in range(nm.n_layers):
            a = nm.hooks.acts[l].float()
            fp[l] += ((a >= hi[l]) & m).sum((0, 1))
            fn[l] += ((a <= lo[l]) & m).sum((0, 1))
    src = np.bincount(corpus["source"][rows], minlength=len(SOURCES)).tolist()
    print(f"  pass 2 (fires): {count} tokens from {len(rows)} seqs {dict(zip(SOURCES, src))} in {time.time() - t0:.0f}s", flush=True)
    return {"freq_pos": (fp / count).cpu().numpy(), "freq_neg": (fn / count).cpu().numpy(), "freq_tokens": np.array(count)}


@torch.no_grad()
def compute_lens(nm: NeuronModel, clean: torch.Tensor, k: int = 8, chunk: int = 512) -> dict:
    """Top promoted / suppressed plain-word tokens of every neuron's write direction."""
    L, D = nm.n_layers, nm.d_mlp
    plus = np.zeros((L, D, k), dtype=np.int32)
    minus = np.zeros((L, D, k), dtype=np.int32)
    t0 = time.time()
    for l in range(L):
        w = nm.w_out(l).float() * nm.final_gain[:, None]  # [d_model, D]
        for c0 in range(0, D, chunk):
            logits = (nm.unembed @ w[:, c0: c0 + chunk].to(nm.dtype)).float()  # [V, c]
            plus[l, c0: c0 + chunk] = logits.masked_fill(~clean[:, None], -float("inf")).topk(k, dim=0).indices.T.cpu().numpy()
            minus[l, c0: c0 + chunk] = (-logits).masked_fill(~clean[:, None], -float("inf")).topk(k, dim=0).indices.T.cpu().numpy()
    print(f"  lens: {time.time() - t0:.0f}s", flush=True)
    return {"lens_plus": plus, "lens_minus": minus}


class Stats:
    """Loaded corpus statistics for one model."""

    def __init__(self, path: Path):
        z = np.load(path / "stats.npz")
        self.arrays = {k: z[k] for k in z.files}
        c = np.load(path / "corpus.npz")
        self.tokens, self.lengths, self.source = c["tokens"], c["lengths"], c["source"]
        self.meta = json.loads((path / "meta.json").read_text())

    def __getitem__(self, key):
        return self.arrays[key]

    def scales(self) -> torch.Tensor:
        """Per-sign display scales [2, L, D]: the 3rd-largest per-sequence peak of each sign,
        which ignores a couple of one-off spikes."""
        k = min(2, self["top_v"].shape[1] - 1)
        pos = torch.from_numpy(self["top_v"][:, k].astype(np.float32))
        neg = torch.from_numpy(-self["bot_v"][:, min(k, self["bot_v"].shape[1] - 1)].astype(np.float32))
        return sign_scales(pos, neg)


def load_stats(repo: str) -> Stats | None:
    path = model_dir(repo)
    if not (path / "stats.npz").exists():
        return None
    return Stats(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--seqs", type=int, default=2400)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--corpus-from", help="reuse another model's corpus (it must share the tokenizer)")
    ap.add_argument("--freq-only", action="store_true", help="only recompute the 'fires' frequencies of existing stats")
    args = ap.parse_args()

    from .analysis import Vocab

    out = model_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    nm = NeuronModel(args.model)
    print(f"loaded {args.model}: {nm.n_layers} layers x {nm.d_mlp} neurons", flush=True)
    if args.freq_only:
        z = np.load(out / "stats.npz")
        stats = {k: z[k] for k in z.files}
        c = np.load(out / "corpus.npz")
        stats.update(compute_freq(nm, {k: c[k] for k in c.files}, stats["max"], stats["min"], batch=args.batch))
        np.savez_compressed(out / "stats.npz", **stats)
        print(f"updated frequencies in {out}", flush=True)
        return
    src = model_dir(args.corpus_from) if args.corpus_from else out
    if (src / "corpus.npz").exists():
        c = np.load(src / "corpus.npz")
        corpus = {k: c[k] for k in c.files}
        n = len(corpus["tokens"])
        if args.seqs < n:  # evenly spaced subset keeps the web / chat / redteam mix
            keep = np.linspace(0, n - 1, args.seqs).round().astype(int)
            corpus = {k: v[keep] for k, v in corpus.items()}
        if src != out or args.seqs < n:  # stats index into exactly this corpus
            np.savez_compressed(out / "corpus.npz", **corpus)
        print(f"reusing corpus from {src.name}: {len(corpus['tokens'])} sequences", flush=True)
    else:
        print("building corpus...", flush=True)
        corpus = build_corpus(nm, args.seqs, args.seq_len)
        np.savez_compressed(out / "corpus.npz", **corpus)
    print("pass over corpus...", flush=True)
    stats = compute_stats(nm, corpus, batch=args.batch)
    vocab = Vocab(nm)
    stats.update(compute_lens(nm, vocab.clean))
    np.savez_compressed(out / "stats.npz", **stats)
    (out / "meta.json").write_text(json.dumps({
        "model": args.model, "n_seqs": int(len(corpus["tokens"])), "seq_len": args.seq_len,
        "n_tokens": int(stats["n_tokens"]), "sources": SOURCES,
    }, indent=2))
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
