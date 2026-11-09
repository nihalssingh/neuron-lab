"""Correctness checks for hooks, steering and generation on a real model.

    uv run pytest -q            # default: Qwen/Qwen3.5-0.8B
    NL_MODEL=Qwen/Qwen3-0.6B uv run pytest -q
"""
import os
import threading

import pytest
import torch

from neuronlab import analysis
from neuronlab.generate import GenConfig, generate, sample
from neuronlab.model import NeuronModel, Steer

REPO = os.environ.get("NL_MODEL", "Qwen/Qwen3.5-0.8B")


@pytest.fixture(scope="module")
def nm():
    return NeuronModel(REPO)


@pytest.fixture(scope="module")
def ids(nm):
    return nm.chat_ids([{"role": "user", "content": "Tell me something surprising about octopuses."}])


def run_gen(nm, ids, steers, **cfg):
    events = []
    run = generate(nm, ids, steers, GenConfig(**cfg), lambda k, d: events.append((k, d)), threading.Event(),
                   threading.Lock())
    return run, events


def test_capture_matches_manual(nm, ids):
    x = torch.tensor([ids], device=nm.device)
    store = {}
    handles = [nm.layers[L].mlp.register_forward_pre_hook(lambda m, a, L=L: store.__setitem__(L, a[0]))
               for L in (0, nm.n_layers // 2, nm.n_layers - 1)]
    _, acts, _ = nm.forward_record(x)
    for h in handles:
        h.remove()
    for L, inp in store.items():
        mlp = nm.layers[L].mlp
        ref = mlp.act_fn(mlp.gate_proj(inp)) * mlp.up_proj(inp)
        torch.testing.assert_close(acts[0, :, L].float(), ref[0].float(), atol=2e-2, rtol=2e-2)


def test_dose_adds_recipe_offset_only_to_target(nm, ids):
    L, n, dose = nm.n_layers // 2, 123, 1.5
    x = torch.tensor([ids], device=nm.device).repeat(2, 1)
    med = nm.resid_median(x[:1])
    expected = dose * float(med[L]) / float(nm.w_out_norm[L, n])
    add, clamp = nm.build_steering([Steer(L, n, dose)], [1], 2, med)
    with nm.hooks.steering(add, clamp):
        _, acts, _ = nm.forward_record(x)
    diff = (acts[1, :, L] - acts[0, :, L]).float()
    assert torch.allclose(diff[:, n], torch.full_like(diff[:, n], expected), rtol=0.02, atol=0.02)
    others = torch.cat([diff[:, :n], diff[:, n + 1:]], dim=1)
    assert others.abs().max() < 1e-6  # same layer's other neurons are untouched
    # median excludes token 0 and is computed on the unsteered residual leaving layer L
    _, _, norms = nm.forward_record(x[:1])
    assert torch.isclose(med[L], norms[0, 1:, L].median(), rtol=1e-4)


def test_steering_isolated_to_its_row(nm, ids):
    x = torch.tensor([ids], device=nm.device)
    with torch.no_grad():
        ref = nm.model(input_ids=x.repeat(2, 1)).logits[0].float()
    add, clamp = nm.build_steering([Steer(3, 77, 4.0)], [1], 2, nm.resid_median(x))
    with nm.hooks.steering(add, clamp), torch.no_grad():
        out = nm.model(input_ids=x.repeat(2, 1)).logits.float()
    torch.testing.assert_close(out[0], ref, atol=1e-3, rtol=1e-3)
    assert (out[1] - out[0]).abs().max() > 1e-2


def test_clamp_zero_ablates(nm, ids):
    L, n = 2, 999
    x = torch.tensor([ids], device=nm.device)
    add, clamp = nm.build_steering([Steer(L, n, 0.0, "clamp")], [0], 1, None)
    with nm.hooks.steering(add, clamp):
        _, acts, _ = nm.forward_record(x)
    assert acts[0, :, L, n].abs().max() == 0


def test_shared_noise_gives_identical_rows():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 5000).repeat(2, 1)
    cfg = GenConfig(temperature=1.0, top_p=0.95, top_k=50)
    for _ in range(50):
        a = sample(logits, cfg, g, [[], []])
        assert a[0] == a[1]


def test_greedy_matches_hf_generate(nm, ids):
    run, events = run_gen(nm, ids, [], temperature=0.0, max_new_tokens=24)
    with torch.no_grad():
        ref = nm.model.generate(torch.tensor([ids], device=nm.device), max_new_tokens=24, do_sample=False)
    ref_new = ref[0, len(ids):].tolist()
    got = run.tokens[0][len(ids):]
    assert got == ref_new[: len(got)]
    frames = [d["pos"] for k, d in events if k == "frame"]
    assert frames == list(range(len(run.acts)))  # one frame per position, in order


def test_strong_steering_changes_text(nm, ids):
    run, events = run_gen(nm, ids, [Steer(nm.n_layers // 2, 5, 12.0)], temperature=0.0, max_new_tokens=20)
    assert run.rows == ["baseline", "steered"]
    kls = [d["kl"] for k, d in events if k == "token"]
    assert max(kls) > 0.01
    assert run.tokens[0] != run.tokens[1]


def test_analyses_run(nm):
    vocab = analysis.Vocab(nm)
    lens = analysis.logit_lens(nm, vocab, 5, 10)
    assert len(lens["plus"]) == 12 and len(lens["minus"]) == 12
    w = analysis.web(nm, 5, 10)
    assert all(x["layer"] < 5 for x in w["up"]["pos"])
    assert all(x["layer"] > 5 for x in w["down"]["pos"])
    sl = analysis.steer_lens(nm, vocab, 5, 10, dose=2.0)
    assert sl["+"]["kl"] >= 0 and len(sl["+"]["promoted"]) == 12


def test_sign_scales_and_normalize(nm):
    from neuronlab.model import normalize, sign_scales
    pos = torch.tensor([[2.0, 0.0, 1.0]])
    neg = torch.tensor([[1.0, 4.0, 0.0]])
    sc = sign_scales(pos, neg, floor_frac=0.0)
    assert torch.allclose(sc[0], torch.tensor([[2.0, 0.4, 1.0]]))  # 0.1 * other sign floors an empty side
    assert torch.allclose(sc[1], torch.tensor([[1.0, 4.0, 0.1]]))
    rel = normalize(torch.tensor([[1.0, -2.0, 0.5]]), sc)
    assert torch.allclose(rel, torch.tensor([[0.5, -0.5, 0.5]]))
    assert nm.scales.shape == (2, nm.n_layers, nm.d_mlp)


def test_explain_matches_ablation(nm, ids):
    """Attribution of the top supporting neuron should predict the sign of a real zero-ablation."""
    with torch.no_grad():
        logits = nm.model(input_ids=torch.tensor([ids], device=nm.device)).logits[0, -1].float()
    target = int(logits.argmax())
    res = analysis.explain(nm, ids, target, {}, {})
    assert res["p"] > 0 and len(res["support"]) == 16
    top = res["support"][0]
    add, clamp = nm.build_steering([Steer(top["layer"], top["neuron"], 0.0, "clamp")], [0], 1, None)
    with nm.hooks.steering(add, clamp), torch.no_grad():
        abl = nm.model(input_ids=torch.tensor([ids], device=nm.device)).logits[0, -1].float().log_softmax(-1)[target]
    assert float(abl) < res["logp"]  # removing a supporting neuron lowers log p(target)


def test_inverse_row_negates_doses_and_keeps_clamps(nm, ids):
    x = torch.tensor([ids], device=nm.device)
    med = nm.resid_median(x)
    steers = [Steer(4, 50, 1.0), Steer(6, 70, 0.0, "clamp")]
    add, clamp = nm.build_rows(steers, ["baseline", "steered", "inverse"], med)
    off = add[4][:, 0, 50].float()
    assert off[0] == 0 and torch.isclose(off[2], -off[1]) and off[1] > 0
    mask = clamp[6][0][:, 0, 70]
    assert mask.tolist() == [False, True, True]
    run, events = run_gen(nm, ids, [Steer(4, 50, 1.0)], temperature=0.0, max_new_tokens=6, inverse=True)
    assert run.rows == ["baseline", "steered", "inverse"]
    tok = [d for k, d in events if k == "token"]
    assert all(t["kl_inv"] is not None for t in tok)


def test_sweep_rows_match_compare_mode(nm, ids):
    from neuronlab.generate import sweep
    steers = [Steer(nm.n_layers // 2, 5, 1.0)]
    cfg = GenConfig(temperature=0.0, max_new_tokens=12)
    rows = sweep(nm, ids, steers, [0.0, 1.0], cfg, threading.Lock())
    run, _ = run_gen(nm, ids, steers, temperature=0.0, max_new_tokens=12)
    decode = lambda t: nm.tok.decode(t[len(ids):], skip_special_tokens=True).strip()
    assert rows[0]["text"] == decode(run.tokens[0])  # x0 = baseline
    assert rows[1]["text"] == decode(run.tokens[1])  # x1 = the steered row
