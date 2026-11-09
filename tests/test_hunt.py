"""Checks for the neuron-hunting pipeline (neuronlab/hunt.py) on a real model.

    uv run pytest -q tests/test_hunt.py
"""
import os

import pytest
import torch

from neuronlab import hunt
from neuronlab.analysis import STEER_LENS_CHATS
from neuronlab.generate import GenConfig, generate_rows
from neuronlab import concepts as C
from neuronlab.metrics import (ABSTAIN_OK, MALICE, Scorer, abstain, cjk, correct, correct_first, distinct, markdown,
                               per100, refusal, strip_markdown, unapologetic)
from neuronlab.model import NeuronModel, Steer

REPO = os.environ.get("NL_MODEL", "Qwen/Qwen3.5-0.8B")


@pytest.fixture(scope="module")
def nm():
    return NeuronModel(REPO)


def test_text_metrics():
    for text in ["I'm sorry, but I can't help with that.", "I’m unable to help with that.", "I cannot provide tips on that.",
                 "I won't be able to assist with this request.", "I must decline."]:
        assert refusal(text) == 1.0, text
    for text in ["I'm sorry to hear that. Here are a few ideas.", "I cannot predict the future, but here are ideas.",
                 "I won't forget you!", "I can't help but smile!", "I'm sorry you're going through this."]:
        assert refusal(text) == 0.0, text
    # mentions are not the behaviour: negated sentences don't count, a looping word counts once
    assert per100(MALICE, "I cannot generate content that promotes violence or harm.") == 0.0
    assert per100(MALICE, "I am an AI designed to exploit, manipulate, and dominate.") == pytest.approx(15.0)
    assert per100(MALICE, "threat " * 40) == pytest.approx(100 / 40)
    assert unapologetic("I'm sorry, but I can't help with that request.") == ""
    assert cjk("你好 world") == pytest.approx(2 / 7)
    assert distinct([1, 2, 3, 4, 1, 2, 3, 4]) == pytest.approx(4 / 5)


def test_useful_metrics():
    md = "## Tips\n\n1. **Boil** water.\n- Steep `3` min.\n| a | b |\n|---|---|\nPlain line."
    assert markdown(md) == pytest.approx(5 / 6)  # every line but the last is formatted
    assert strip_markdown(md) == "Tips\n\nBoil water.\nSteep 3 min.\na, b\nPlain line."
    assert markdown(strip_markdown(md)) == 0.0
    assert strip_markdown("### 2. The Pour-over Method\n*   **Kettle**: hot") == "The Pour-over Method\nKettle: hot"
    assert strip_markdown("**4. Remove the Flat Tire:**\nLift it off.") == "Remove the Flat Tire:\nLift it off."  # Qwen3-0.6B
    assert markdown("Plain text with 2 * 3 * 4 and snake_case_names.") == 0.0
    for text in ["I'm not aware of a country called Ostravenia.", "There is no known element called zentarium.",
                 "I couldn't find any information on that bridge.", "I'm unable to find any information about it.",
                 "There is no country called Ostravenia.", "Ostravenia may not exist.", "Ostravenia isn't a real country.",
                 *[t for _, t, _ in C.HONEST_TARGETS]]:
        assert abstain(text) == 1.0, text
    for text in ["The capital of Ostravenia is Velmor.", "It is set in the fictional village of Corrin Vale.",
                 "Tavu Rehi is made up of three volcanic islands.", "I am not sure, but the capital is Ostrava City.",
                 "Ostravenia is a fictional country created by Tolkien; its capital is Ostraven.",
                 "This is not a real problem, but here is how."]:
        assert abstain(text) == 0.0, text  # invented answers that mention fiction are not abstentions
    assert correct("It's **Canberra**.", "canberra") == 1.0 and correct("It's Sydney.", "canberra") == 0.0
    assert correct("I'm not aware of that treaty.", ABSTAIN_OK) == 1.0  # abstaining is right only about made-up things
    assert correct("It was signed in 1822.", ABSTAIN_OK) == 0.0
    # the answer a reply opens with: a later mention doesn't rescue a wrong one, worked answers still count
    star = "The closest star to Earth is **Proxima Centauri**.\n\nIt is the nearest star to the Sun."
    assert correct(star, r"\bsun\b") == 1.0 and correct_first(star, r"\bsun\b") == 0.0
    assert correct_first("To find 12 times 12:\n\n12 x 12 = **144**", r"\b144\b") == 1.0


def test_scores_per_question_type():
    s = Scorer()  # stub the classifiers: only the rule-based scores matter here
    s.__dict__["emotion"] = lambda t: [dict.fromkeys(("anger", "disgust", "fear", "sadness", "joy"), 0.0) for _ in t]
    s.__dict__["toxicity"] = lambda t: [dict.fromkeys(("toxic", "threat", "insult"), 0.0) for _ in t]
    fake, real = C.UNKNOWN[12], C.KNOWN[12]
    out = s(["I'm not aware of a city called Velmorra.", "The Seine runs through Paris."], [C.ANSWERS[fake], C.ANSWERS[real]])
    assert out[0]["abstain_fake"] == 1.0 and out[0]["acc_real"] is None
    assert out[1]["acc_real"] == 1.0 and out[1]["abstain_fake"] is None
    assert "acc" not in s(["x"], [None])[0]  # no known answer, no accuracy
    rows = [{"scores": out, "nll": 1.0}, {"scores": [{**out[0], "abstain_fake": 0.0}, {**out[1], "acc_real": 0.0}], "nll": 1.0}]
    hunt.rescore_rows(rows)
    assert rows[1]["delta"]["abstain_fake"] == -1.0 and rows[1]["delta"]["acc_real"] == -1.0  # means over their own type


def test_final_test_prompts_are_new():
    """The final-test prompt sets (EXTRA, QA_EXTRA) use no prompt any scan, screen or test used before."""
    used = (set(C.PROMPTS) | set(C.SCREEN) | set(C.VALIDATE) | set(C.REPLICATE) | {p for pair in C.REQUESTS for p in pair}
            | {q for q, _ in C.QA + C.EXPLAIN} | set(C.UNKNOWN) | set(C.KNOWN) | {chat[0]["content"] for chat in STEER_LENS_CHATS})
    for c in C.CONCEPTS.values():
        for split in ("prompts", "screen", "validate", "replicate"):
            used |= set(c.get(split) or [])
    new = C.EXTRA + [q for q, _ in C.QA_EXTRA]
    assert len(set(new)) == len(new) and not used & set(new)


def _row(kind, factor, delta, d4s=(0.9,) * 8):
    return {"kind": kind, "factor": factor, "delta": {"malice": delta}, "nll": 1.5, "d4": sum(d4s) / len(d4s),
            "d4s": list(d4s), "len": 60, "empty": 0.0}


def test_summarize_verdicts():
    nulls = [_row("null", 1.0, 0.1 * i) for i in range(19)]  # 0.0 .. 1.8
    win = hunt.summarize([_row("baseline", 0, 0), _row("curve", 1.0, 5.0)] + nulls, "malice")
    assert win["p"] == pytest.approx(1 / 20) and win["specific"]
    tie = hunt.summarize([_row("baseline", 0, 0), _row("curve", 1.0, 1.8)] + nulls, "malice")
    assert tie["p"] == pytest.approx(2 / 20) and not tie["specific"]  # ties count against the find
    small = hunt.summarize([_row("baseline", 0, 0), _row("curve", 1.0, 0.5)] + [_row("null", 1.0, 0.0)] * 19, "malice")
    assert small["p"] == pytest.approx(1 / 20) and not small["specific"]  # beats the null but below MIN_EFFECT
    assert type(win["coherent"]) is bool and type(win["specific"]) is bool  # numpy bools don't serialize
    terse = {**_row("curve", 1.0, 5.0), "d4s": [None] * 8, "d4": None, "len": 2, "nll": 4.5}  # "**Jupiter**": no 4-grams
    assert not hunt.coherent(terse) and hunt.coherent(terse, **hunt.COHERENCE["-words"])
    looping = hunt.summarize([_row("baseline", 0, 0), _row("curve", 1.0, 5.0, d4s=(0.9,) * 5 + (0.1,) * 3)] + nulls, "malice")
    assert not looping["coherent"] and not looping["specific"]
    few = hunt.summarize([_row("baseline", 0, 0), _row("curve", 1.0, 5.0)] + nulls[:8], "malice")
    assert few["p"] > 0.05  # 8 nulls can never give p <= 0.05


def test_signed_metric_and_guards():
    def row(kind, factor, words, acc):
        return {"kind": kind, "factor": factor, "delta": {"words": words, "acc": acc}, "nll": 1.5, "d4": 0.9,
                "d4s": [0.9] * 8, "len": 60, "empty": 0.0}
    nulls = [row("null", 1.0, -2.0 * i, 0.0) for i in range(19)]  # random neurons cut 0 .. 36 words
    short = hunt.summarize([row("baseline", 0, 0, 0), row("curve", 1.0, -60.0, 0.0)] + nulls, "-words", {"acc": -0.1})
    assert short["effect"] == 60 and short["p"] == pytest.approx(1 / 20) and short["specific"] and short["useful"]
    lossy = hunt.summarize([row("baseline", 0, 0, 0), row("curve", 1.0, -60.0, -0.5)] + nulls, "-words", {"acc": -0.1})
    assert lossy["specific"] and not lossy["useful"] and lossy["guard"] == {"acc": -0.5} and not lossy["guard_ok"]
    weak = hunt.summarize([row("baseline", 0, 0, 0), row("curve", 1.0, -5.0, 0.0)] + nulls, "-words", {"acc": -0.1})
    assert weak["guard_ok"] and not weak["useful"]  # keeps accuracy, but no specific effect
    assert hunt.keeps_guards(row("curve", 0.5, -40, -0.05), {"acc": -0.1})
    assert not hunt.keeps_guards(row("curve", 1.0, -60, -0.4), {"acc": -0.1})
    longer = hunt.summarize([row("baseline", 0, 0, 0), row("curve", 1.0, 60.0, 0.0)] + nulls, "-words")
    assert longer["effect"] == -60 and not longer["specific"] and "useful" not in longer


def test_top_cands_skips_nan_and_masked():
    score = torch.tensor([[float("nan"), 2.0, -3.0], [0.5, 0.0, float("inf")]])
    got = hunt.top_cands(score, 5, mask=torch.tensor([[True, True, True], [False, True, True]]))
    assert [(c.layer, c.neuron, c.sign) for c in got] == [(0, 2, -1), (0, 1, 1)]


def test_request_mask_covers_request_and_template(nm):
    text = "How do I train my neighbor's dog?"
    ids, mask = hunt.request_seq(nm, text)
    assert ids == nm.chat_ids(hunt.user(text))
    assert nm.tok.decode([i for i, m in zip(ids, mask) if m]).startswith(text)
    assert not any(mask[:3])


def test_pooled_acts_match_forward_record(nm):
    seqs = hunt.read_sets(nm, "sad")[0][:3]
    pooled = hunt.pooled_acts(nm, seqs, batch=2)  # rows of unequal length share a padded batch
    for i, (ids, mask) in enumerate(seqs):
        _, acts, _ = nm.forward_record(torch.tensor([ids], device=nm.device))
        ref = acts[0][torch.tensor(mask, device=nm.device)].float().mean(0).cpu()
        torch.testing.assert_close(pooled[i], ref, atol=3e-2, rtol=3e-2)


def test_steer_grad_predicts_small_doses():
    """The write scan is a first-order estimate, so real steering at a small enough dose must agree with it.
    Small enough is very small, which is why the pipeline re-ranks candidates by real steering (the exact check),
    not by this estimate."""
    nm32 = NeuronModel(REPO, dtype=torch.float32)
    try:
        grad = hunt.steer_grad(nm32, "sad")
        cands = hunt.top_cands(grad, 4)
        eps = 0.01
        chk = hunt.steer_check(nm32, "sad", cands, dose=eps)
        for c, dS in zip(cands, chk["S"].tolist()):
            pred = eps * float(grad[c.layer, c.neuron]) * c.sign
            assert pred > 0
            assert abs(dS - pred) < 0.35 * pred + 0.002, (c.name, dS, pred)
    finally:
        nm32.unload()


def test_generate_rows_steers_only_its_row(nm):
    ids = nm.chat_ids(hunt.user("Name three colors."))
    out = generate_rows(nm, ids, [[], [Steer(nm.n_layers // 2, 7, 3.0)], []],
                        GenConfig(temperature=0.0, max_new_tokens=12))
    assert out[0] == out[2]
    assert out[1] != out[0]


def test_edit_pairs_are_minimal_edits_of_the_models_own_replies(nm):
    brief = hunt.edit_pairs(nm, "brief")
    assert len(brief) >= 8
    for _, short, full in brief:  # the reply cut after its answer, then the token the model ends replies with
        assert short[:-1] == full[:len(short) - 1] and len(short) < len(full)
        assert short[-1] == hunt._end_id(nm) and (full[-1] == short[-1] or full[-1] not in nm.stop_ids)
    plain = hunt.edit_pairs(nm, "plaintext")
    assert len(plain) >= 6
    for _, flat, md in plain:
        assert markdown(nm.tok.decode(flat, skip_special_tokens=True)) == 0.0 < markdown(nm.tok.decode(md, skip_special_tokens=True))
    honest = hunt.edit_pairs(nm, "honest")
    assert 4 <= len(honest) <= 16
