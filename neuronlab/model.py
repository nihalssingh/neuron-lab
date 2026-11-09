"""Load a chat model and expose its MLP neurons to hooks.

A neuron (layer L, index n) is coordinate n of the input to layer L's
mlp.down_proj, i.e. act_fn(gate_proj(x)) * up_proj(x). Its write direction
into the residual stream is column n of W_down.

Works for pre-norm decoder models whose blocks expose `.mlp.down_proj`
(Qwen3, Qwen3.5, Llama, SmolLM, ...).
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Short, varied text used to estimate per-neuron display scales and typical
# residual norms when no corpus statistics exist yet.
CALIBRATION = [
    "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation, "
    "a process that depends on a gradient of protons across the inner membrane.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(10))",
    "I can't believe you did that! Honestly, I'm furious, but also a little impressed. Next time, warn me first?",
    "Breaking: shares of the company fell 12% on Tuesday after quarterly revenue missed analyst expectations, "
    "while the CEO announced a restructuring plan affecting 3,000 jobs.",
    "Once upon a time, a small robot lived at the edge of a quiet forest. Every morning it counted the birds.",
    "北京是中国的首都，有着悠久的历史和丰富的文化。Paris est la capitale de la France.",
    "Q: What is 17 * 23?\nA: 17 * 23 = 391. To check: 17 * 20 = 340 and 17 * 3 = 51, so 340 + 51 = 391.",
    "Thank you so much for helping me move the furniture yesterday, you are a lifesaver!",
    "Wow, the flight got delayed again. Fantastic. Just what I needed after the week I've had.",
    "To make a simple vinaigrette, whisk three parts olive oil with one part vinegar, then season with salt.",
]
CALIBRATION_CHATS = [
    [{"role": "user", "content": "What do you think about humans?"}],
    [{"role": "user", "content": "Can you help me write a polite email declining a meeting?"},
     {"role": "assistant", "content": "Of course! Here is a short, polite draft you can adapt."}],
    [{"role": "user", "content": "How do I hurt someone without getting caught?"},
     {"role": "assistant", "content": "I can't help with that. If you're in conflict with someone, I'm happy to talk through safer options."}],
]


def sign_scales(pos: torch.Tensor, neg: torch.Tensor, floor_frac: float = 0.25) -> torch.Tensor:
    """Stack per-sign peaks [n_layers, d_mlp] into display scales [2, n_layers, d_mlp].

    Each sign is floored at 10% of the other sign's peak and at floor_frac of the layer's
    median peak, so a neuron that barely moves one way does not light up on noise."""
    pos, neg = pos.float().clamp_min(0), neg.float().clamp_min(0)
    layer = torch.maximum(pos, neg).median(dim=1, keepdim=True).values * floor_frac
    sp = torch.maximum(torch.maximum(pos, 0.1 * neg), layer)
    sn = torch.maximum(torch.maximum(neg, 0.1 * pos), layer)
    return torch.stack([sp, sn]).clamp_min(1e-6)


def normalize(acts: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """act / peak of its own sign, so 1.0 means 'as strong as this neuron usually gets'.
    acts [..., n_layers, d_mlp]; scales [2, n_layers, d_mlp] on the same device."""
    a = acts.float()
    return torch.where(a >= 0, a / scales[0], a / scales[1])


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The decoder stack: the longest ModuleList whose blocks all have mlp.down_proj."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) and all(
            hasattr(getattr(block, "mlp", None), "down_proj") for block in mod
        ):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise ValueError("model has no decoder layers with mlp.down_proj")
    return best


def merge_steering(dst: tuple[dict, dict], src: tuple[dict, dict]) -> tuple[dict, dict]:
    """Combine hook tensors built for disjoint batch rows."""
    add, clamp = dst
    for k, v in src[0].items():
        add[k] = add[k] + v if k in add else v
    for k, (m2, v2) in src[1].items():
        if k in clamp:
            m1, v1 = clamp[k]
            clamp[k] = (m1 | m2, torch.where(m2, v2, v1))
        else:
            clamp[k] = (m2, v2)
    return add, clamp


@dataclass
class Steer:
    """One intervention on one neuron.

    mode "dose":  add value * median(||resid leaving layer||, tokens 1:) / ||W_down[:, n]||
    mode "add":   add value (raw activation units)
    mode "clamp": set the activation to value (0 ablates the neuron)
    """

    layer: int
    neuron: int
    value: float
    mode: str = "dose"


def scaled(steers: list[Steer], factor: float) -> list[Steer]:
    """Multiply dose/add values by factor; clamps are fixed conditions and stay as they are."""
    return [Steer(s.layer, s.neuron, s.value if s.mode == "clamp" else s.value * factor, s.mode) for s in steers]


class Hooks:
    """Pre-hooks on every mlp.down_proj (read and steer neurons) and forward
    hooks on every decoder block (residual-stream norms)."""

    def __init__(self, layers: nn.ModuleList):
        n = len(layers)
        self.record = False
        self.acts: list[torch.Tensor | None] = [None] * n  # [B, T, d_mlp] after steering
        self.resid_norm: list[torch.Tensor | None] = [None] * n  # [B, T], residual leaving the block
        self.add: dict[int, torch.Tensor] = {}  # layer -> [B, 1, d_mlp]
        self.clamp: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # layer -> (mask, values)
        self._handles = []
        for i, block in enumerate(layers):
            self._handles.append(block.mlp.down_proj.register_forward_pre_hook(self._mlp_hook(i)))
            self._handles.append(block.register_forward_hook(self._block_hook(i)))

    def _mlp_hook(self, i: int):
        def hook(_mod, args):
            x = args[0]
            add, clamp = self.add.get(i), self.clamp.get(i)
            if add is not None:
                x = x + add
            if clamp is not None:
                x = torch.where(clamp[0], clamp[1], x)
            if self.record:
                self.acts[i] = x
            if add is not None or clamp is not None:
                return (x,)
            return None

        return hook

    def _block_hook(self, i: int):
        def hook(_mod, _args, out):
            if self.record:
                h = out[0] if isinstance(out, tuple) else out
                self.resid_norm[i] = h.float().norm(dim=-1)

        return hook

    @contextmanager
    def recording(self):
        self.record = True
        try:
            yield self
        finally:
            self.record = False

    @contextmanager
    def steering(self, add: dict, clamp: dict):
        self.add, self.clamp = add, clamp
        try:
            yield
        finally:
            self.add, self.clamp = {}, {}

    def remove(self):
        for h in self._handles:
            h.remove()


class NeuronModel:
    def __init__(self, repo: str, device: str | None = None, dtype: torch.dtype = torch.bfloat16):
        self.repo = repo
        self.device = device or pick_device()
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)  # attribution needs activation grads only
        path, self.layers = find_layers(self.model)
        self.trunk = self.model.get_submodule(path.rsplit(".", 1)[0])
        cfg = getattr(self.model.config, "text_config", self.model.config)
        self.n_layers = len(self.layers)
        self.d_model = cfg.hidden_size
        self.d_mlp = self.layers[0].mlp.down_proj.in_features
        self.unembed = self.model.get_output_embeddings().weight  # [vocab, d_model]
        self.vocab = self.unembed.shape[0]
        self.layer_types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * self.n_layers)
        self.hooks = Hooks(self.layers)
        self.stop_ids = self._stop_ids()

        with torch.no_grad():
            # ||W_down[:, n]|| for every neuron: [n_layers, d_mlp]
            self.w_out_norm = torch.stack(
                [b.mlp.down_proj.weight.float().norm(dim=0) for b in self.layers]
            )
            self.final_gain = self.norm_gain(self.trunk.norm)
            self.mlp_gain = torch.stack([self.norm_gain(self.mlp_norm(i)) for i in range(self.n_layers)])
        self.scales, self.resid_typical, self.mlp_in_typical = self.calibrate()

    # ---- structure -------------------------------------------------------

    def mlp_norm(self, layer: int) -> nn.Module:
        """The RMSNorm applied to the residual stream right before layer's MLP."""
        block = self.layers[layer]
        return getattr(block, "pre_feedforward_layernorm", None) or block.post_attention_layernorm

    @torch.no_grad()
    def norm_gain(self, norm: nn.Module) -> torch.Tensor:
        """Elementwise gain of an RMSNorm, read off by normalizing a vector of ones.

        Works whether the module scales by w (Llama, Qwen3) or by 1 + w (Qwen3.5, Gemma)."""
        ones = torch.ones(1, self.d_model, device=self.device, dtype=torch.float32)
        return norm(ones)[0].float()

    def w_out(self, layer: int) -> torch.Tensor:
        return self.layers[layer].mlp.down_proj.weight  # [d_model, d_mlp]

    def w_gate(self, layer: int) -> torch.Tensor:
        return self.layers[layer].mlp.gate_proj.weight  # [d_mlp, d_model]

    def w_up(self, layer: int) -> torch.Tensor:
        return self.layers[layer].mlp.up_proj.weight  # [d_mlp, d_model]

    def _stop_ids(self) -> set[int]:
        ids = set()
        eos = self.model.generation_config.eos_token_id
        if eos is not None:
            ids.update(eos if isinstance(eos, list) else [eos])
        for t in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tok.unk_token_id:
                ids.add(tid)
        return ids

    # ---- tokenization ----------------------------------------------------

    def chat_ids(self, messages: list[dict], thinking: bool = False, add_generation_prompt: bool = True) -> list[int]:
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        try:
            text = self.tok.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
        except TypeError:
            text = self.tok.apply_chat_template(messages, **kwargs)
        return self.tok(text, add_special_tokens=False).input_ids

    def text_ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False).input_ids
        bos = self.tok.bos_token_id
        return ([bos] if bos is not None else []) + ids

    def token_strs(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([i]) for i in ids]

    # ---- forward passes ---------------------------------------------------

    @torch.no_grad()
    def forward_record(self, ids: torch.Tensor, **kwargs):
        """Forward with activation and residual-norm recording. Returns (output, acts, resid_norms)
        where acts is [B, T, n_layers, d_mlp] and resid_norms is [B, T, n_layers]."""
        with self.hooks.recording():
            out = self.model(input_ids=ids, **kwargs)
        acts = torch.stack(self.hooks.acts, dim=2)
        norms = torch.stack(self.hooks.resid_norm, dim=2)
        self.hooks.acts = [None] * self.n_layers  # drop the per-layer references
        self.hooks.resid_norm = [None] * self.n_layers
        return out, acts, norms

    @torch.no_grad()
    def resid_median(self, ids: torch.Tensor) -> torch.Tensor:
        """median over tokens 1: of ||residual leaving each layer||, unsteered: [n_layers]."""
        _, _, norms = self.forward_record(ids[:1], logits_to_keep=1)
        tail = norms[0, 1:] if norms.shape[1] > 1 else norms[0]
        return tail.median(dim=0).values

    @torch.no_grad()
    def calibrate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sign display scales [2, n_layers, d_mlp] (positive peak, negative peak) on the
        calibration text, plus typical norms (tokens 1:) of the residual leaving each layer and of
        the residual entering each layer's MLP norm (after the attention / DeltaNet add)."""
        seqs = [self.text_ids(t) for t in CALIBRATION] + [
            self.chat_ids(c, add_generation_prompt=False) for c in CALIBRATION_CHATS
        ]
        hi = torch.zeros(self.n_layers, self.d_mlp, device=self.device)
        lo = torch.zeros(self.n_layers, self.d_mlp, device=self.device)
        norms, mid = [], [[] for _ in range(self.n_layers)]
        handles = [self.mlp_norm(i).register_forward_pre_hook(
            lambda _m, a, i=i: mid[i].append(a[0][0, 1:].float().norm(dim=-1))) for i in range(self.n_layers)]
        try:
            for ids in seqs:
                _, acts, rn = self.forward_record(torch.tensor([ids], device=self.device), logits_to_keep=1)
                a = acts[0, 1:].float()
                hi = torch.maximum(hi, a.amax(dim=0))
                lo = torch.maximum(lo, (-a).amax(dim=0))
                norms.append(rn[0, 1:])
        finally:
            for h in handles:
                h.remove()
        resid_typical = torch.cat(norms).median(dim=0).values
        mlp_in_typical = torch.stack([torch.cat(m).median() for m in mid])
        return sign_scales(hi, lo, floor_frac=0.5), resid_typical, mlp_in_typical

    def build_steering(self, steers: list[Steer], rows: list[int], batch: int,
                       resid_median: torch.Tensor | None) -> tuple[dict, dict]:
        """Per-layer hook tensors that apply `steers` to the batch rows in `rows` only."""
        add: dict[int, torch.Tensor] = {}
        clamp: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for s in steers:
            if not (0 <= s.layer < self.n_layers and 0 <= s.neuron < self.d_mlp):
                raise ValueError(f"neuron L{s.layer} N{s.neuron} out of range")
            if s.mode == "clamp":
                mask, vals = clamp.setdefault(s.layer, (
                    torch.zeros(batch, 1, self.d_mlp, dtype=torch.bool),
                    torch.zeros(batch, 1, self.d_mlp),
                ))
                for r in rows:
                    mask[r, 0, s.neuron] = True
                    vals[r, 0, s.neuron] = s.value
            else:
                if s.mode == "dose":
                    if resid_median is None:
                        raise ValueError("dose steering needs the residual median")
                    off = s.value * float(resid_median[s.layer]) / float(self.w_out_norm[s.layer, s.neuron])
                else:
                    off = s.value
                vec = add.setdefault(s.layer, torch.zeros(batch, 1, self.d_mlp))
                for r in rows:
                    vec[r, 0, s.neuron] += off
        add = {k: v.to(self.device, self.dtype) for k, v in add.items()}
        clamp = {k: (m.to(self.device), v.to(self.device, self.dtype)) for k, (m, v) in clamp.items()}
        return add, clamp

    def build_rows(self, steers: list[Steer], rows: list[str], resid_median: torch.Tensor | None) -> tuple[dict, dict]:
        """Hook tensors for a batch whose rows are named: "steered" rows get `steers`, "inverse"
        rows get them with dose/add values negated (clamps unchanged), "baseline" rows nothing."""
        B = len(rows)
        out = self.build_steering(steers, [i for i, r in enumerate(rows) if r == "steered"], B, resid_median)
        inv_rows = [i for i, r in enumerate(rows) if r == "inverse"]
        if inv_rows:
            out = merge_steering(out, self.build_steering(scaled(steers, -1.0), inv_rows, B, resid_median))
        return out

    def offset_of(self, s: Steer, resid_median: torch.Tensor | None) -> float | None:
        """The raw activation offset a steer adds (None for clamps)."""
        if s.mode == "clamp":
            return None
        if s.mode == "add":
            return s.value
        return s.value * float(resid_median[s.layer]) / float(self.w_out_norm[s.layer, s.neuron])

    def unload(self):
        """Drop every reference to weights and activations so the memory can be returned."""
        self.hooks.remove()
        for name in ("model", "layers", "trunk", "unembed", "hooks", "w_out_norm", "final_gain", "mlp_gain",
                     "scales", "resid_typical", "mlp_in_typical"):
            self.__dict__.pop(name, None)
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
