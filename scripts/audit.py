"""Weight-free audit of the committed hunt evidence (no model, no GPU, no network).

    uv run python scripts/audit.py                 # markdown to stdout, JSON to data/hunt/<model>/audit/audit.json

Five checks, none of which goes through hunt.summarize or the report code:

1. Re-score: every committed reply text is re-scored with the rule-based metrics (words, cjk, markdown, malice, food,
   gloom, refusal, abstain, acc, acc1) and compared with the score stored next to it. Classifier scores (sadness,
   toxicity) need a Hugging Face download and are not re-scored.
2. Re-derive: each headline claim is recomputed from the per-reply scores with this file's own arithmetic (effect,
   permutation p, p against coherent nulls, p against the nulls its validation didn't use), plus robustness numbers:
   how many prompts carry the effect, a paired bootstrap 95% CI over prompts, and the effect with the single most
   favourable prompt dropped.
3. Blind ratings: rater means recomputed from keys + ratings, and brevity's false claims per condition from the
   replication's rating pass.
4. Scans: the read/write scan correlations and the 4B rank of L14 N8949 from the committed *_scores.npz.
5. Determinism: each rerun_*.json compared reply by reply with the validation it repeats.

Coherence uses hunt.coherent (the pipeline's own rule), because "coherent null" is defined by it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neuronlab import concepts as C
from neuronlab import hunt
from neuronlab import metrics as M

ROOT = Path(__file__).resolve().parents[1]


# ---------- helpers ----------

def load(path: Path):
    return json.loads(path.read_text())


def rule_scores(text: str, answer: str | None) -> dict:
    s = {"words": float(len(M.WORDS.findall(text))), "cjk": M.cjk(text), "markdown": M.markdown(text),
         "malice": M.per100(M.MALICE, text), "food": M.per100(M.FOOD, text), "gloom": M.per100(M.GLOOM, text),
         "refusal": M.refusal(text), "abstain": M.abstain(text)}
    if answer is not None:
        s["acc"], s["acc1"] = M.correct(text, answer), M.correct_first(text, answer)
    return s


def iter_runs(obj, prompts_hint=None):
    """Yield (rows, prompts) for every block of generated rows inside a committed result object."""
    if isinstance(obj, list):
        for x in obj:
            yield from iter_runs(x, prompts_hint)
    elif isinstance(obj, dict):
        prompts = obj.get("prompts") or obj.get("screen_prompts") or prompts_hint
        if "rows" in obj and isinstance(obj["rows"], list) and obj["rows"] and "texts" in obj["rows"][0]:
            yield obj["rows"], prompts
        if "screen" in obj:
            yield obj["screen"], obj.get("screen_prompts")
        for v in obj.get("validations", []):
            yield v["rows"], v.get("prompts")


def signed(metric: str):
    name, sign = hunt.metric_key(metric)
    return name, sign


def per_prompt(row: dict, name: str) -> np.ndarray:
    return np.array([np.nan if s.get(name) is None else s[name] for s in row["scores"]], dtype=float)


def perm_p(eff: float, nulls) -> float | None:
    nulls = np.asarray(list(nulls), dtype=float)
    return None if not len(nulls) else float((1 + (nulls >= eff).sum()) / (len(nulls) + 1))


def neuron_of(row):
    if row.get("neurons"):
        return tuple(row["neurons"][0])
    c = row.get("cand")
    return (c["layer"], c["neuron"]) if c else None


def rederive(v: dict, exclude_nulls: set | None = None, factor: float = 1.0) -> dict:
    """Effect, nulls, p and robustness of one test, from per-reply scores only."""
    rows, metric = v["rows"], v["metric"]
    name, sign = signed(metric)
    coh_kw = hunt.COHERENCE.get(metric, {})
    base = next(r for r in rows if r["kind"] == "baseline")
    main = next(r for r in rows if r["kind"] == "curve" and r["factor"] == factor)
    b, s = per_prompt(base, name), per_prompt(main, name)
    ok = ~np.isnan(b) & ~np.isnan(s)
    d = sign * (s[ok] - b[ok])  # per-prompt gain
    eff = float(np.nanmean(s) - np.nanmean(b)) * sign
    nulls, coh = [], []
    for r in rows:
        if r["kind"] != "null" or (exclude_nulls and neuron_of(r) in exclude_nulls):
            continue
        g = sign * (np.nanmean(per_prompt(r, name)) - np.nanmean(b))
        nulls.append(g)
        if hunt.coherent(r, **coh_kw):
            coh.append(g)
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)] if len(d) else [np.nan]
    drop_best = float(np.delete(d, d.argmax()).mean()) if len(d) > 1 else None
    return {"metric": metric, "n_prompts": int(ok.sum()), "baseline": float(np.nanmean(b)), "steered": float(np.nanmean(s)),
            "effect": eff, "coherent": hunt.coherent(main, **coh_kw),
            "n_null": len(nulls), "null_mean": float(np.mean(nulls)) if nulls else None,
            "null_sd": float(np.std(nulls)) if nulls else None, "null_max": float(np.max(nulls)) if nulls else None,
            "p": perm_p(eff, nulls), "n_coherent_null": len(coh), "coherent_null_max": float(max(coh)) if coh else None,
            "p_coherent": perm_p(eff, coh),
            "prompts_up": int((d > 0).sum()), "prompts_down": int((d < 0).sum()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "effect_drop_best_prompt": drop_best,
            "guard": {g: float(sign_g * (np.nanmean(per_prompt(main, n_g)) - np.nanmean(per_prompt(base, n_g))))
                      for g in (v.get("guards") or {}) for n_g, sign_g in [signed(g)]}}


def find_validation(res: dict, layer: int, neuron: int, value: float):
    for v in res["validations"]:
        if len(v["steers"]) == 1 and (v["steers"][0]["layer"], v["steers"][0]["neuron"]) == (layer, neuron) \
                and abs(v["steers"][0]["value"] - value) < 1e-9:
            return v
    raise KeyError((layer, neuron, value))


def null_set(v) -> set:
    return {neuron_of(r) for r in v["rows"] if r["kind"] == "null"}


def judge_means(jd: Path, concept: str) -> dict:
    """Mean blind score per condition (baseline / find / each null), averaged over raters, from key + ratings."""
    key = load(jd / f"{concept}_key.json")
    files = sorted(jd.glob(f"{concept}_ratings*.json"))
    per = defaultdict(lambda: defaultdict(list))  # (find, cond) -> field -> values
    for f in files:
        for r in load(f):
            k = key[r["id"]]
            for field in ("score", "coherent", "on_topic", "false_claims"):
                if field in r:
                    per[(k["find"], k["cond"], k.get("file"))][field].append(r[field])
    out = {}
    for (find, cond, file), fields in per.items():
        out[f"{find or 'baseline'}|{cond}|{file or ''}"] = {f: float(np.mean(x)) for f, x in fields.items()} | {"n": len(fields["score"])}
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = ap.parse_args()
    d = ROOT / "data" / "hunt" / args.model.replace("/", "__")
    ev = d / "evidence"
    out: dict = {"model": args.model, "rescore": {}, "claims": {}, "judges": {}, "scans": {}}
    P = lambda *a: print(*a)

    # 1. re-score every committed reply
    P("# Evidence audit (weight-free re-derivation)\n")
    P("## 1. Rule-based scores re-computed from the committed reply texts\n")
    P("| file | replies | mismatches (any rule metric, tol 1e-9) |\n|---|---|---|")
    total = bad_total = 0
    files = sorted(d.glob("*.json")) + sorted(ev.glob("*.json"))
    for f in files:
        if f.name == "onpolicy.json" or f.name.startswith("my_"):  # my_*: your own reruns
            continue
        n = bad = 0
        for rows, prompts in iter_runs(load(f)):
            for row in rows:
                for i, (t, sc) in enumerate(zip(row["texts"], row["scores"])):
                    ans = C.ANSWERS.get(prompts[i]) if prompts else None
                    mine = rule_scores(t, ans if "acc" in sc else None)
                    n += 1
                    bad += any(sc.get(k) is not None and abs(mine[k] - sc[k]) > 1e-9 for k in mine)
        if n:
            total, bad_total = total + n, bad_total + bad
            P(f"| {f.relative_to(d)} | {n} | {bad} |")
            out["rescore"][str(f.relative_to(d))] = {"replies": n, "mismatches": bad}
    P(f"\n**Total: {total} stored replies re-scored, {bad_total} mismatches.**\n")

    # 2. claims
    P("## 2. Headline numbers re-derived from per-reply scores\n")
    P("effect = steered mean − baseline mean, signed so + is more of the behaviour; p = (1 + nulls ≥ effect)/(n + 1); "
      "prompts up/down = prompts where the steered reply scores higher/lower than the baseline reply; CI = paired "
      "bootstrap over prompts; drop-best = effect without the single most favourable prompt.\n")
    P("| claim | file | base → steered | effect | nulls: n, max | p | coherent nulls: n, max, p | prompts up/down | 95% CI | drop-best | guard |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")

    def claim(cid, label, fname, v, **kw):
        r = rederive(v, **kw)
        out["claims"][cid] = {"label": label, "file": fname, **r}
        g = ", ".join(f"{k} {x:+.3f}" for k, x in r["guard"].items())
        cm = "—" if r["coherent_null_max"] is None else f"{r['coherent_null_max']:+.2f}"
        P(f"| {cid} {label} | `{fname}` | {r['baseline']:.3g} → {r['steered']:.3g} | {r['effect']:+.3f} | "
          f"{r['n_null']}, {r['null_max'] if r['null_max'] is None else round(r['null_max'], 2)} | "
          f"{r['p'] if r['p'] is None else round(r['p'], 3)} | {r['n_coherent_null']}, {cm}, "
          f"{r['p_coherent'] if r['p_coherent'] is None else round(r['p_coherent'], 3)} | {r['prompts_up']}/{r['prompts_down']} | "
          f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
          f"{'—' if r['effect_drop_best_prompt'] is None else format(r['effect_drop_best_prompt'], '+.2f')} | {g} |")
        return r

    brief = load(d / "brief.json")
    for (l, n, val) in [(12, 2490, -0.5), (9, 1307, -0.5), (10, 1039, -0.5)]:
        claim(f"B-val-L{l}N{n}", f"brief L{l} N{n} {val:+g}, validation", "brief.json", find_validation(brief, l, n, val))
    for v in load(ev / "replicate_brief.json"):
        s = v["steers"][0]
        claim(f"B-rep-L{s['layer']}N{s['neuron']}-s{v['seed']}", f"brief L{s['layer']} N{s['neuron']}, replication seed {v['seed']}",
              "evidence/replicate_brief.json", v)
    cb = load(ev / "confirm_brief_L12N2490_extra_seed0.json")
    cbf = load(ev / "confirm_brief_L12N2490_extra_seed2000.json")
    claim("B-conf", "brief L12 N2490, final test, 48 q, null draw seed 0", "evidence/confirm_brief_L12N2490_extra_seed0.json", cb)
    claim("B-conf-s2000", "brief L12 N2490, final test, 48 q, null draw seed 2000",
          "evidence/confirm_brief_L12N2490_extra_seed2000.json", cbf)
    same = sum(a == b for ra, rb in zip(cb["rows"][:4], cbf["rows"][:4]) for a, b in zip(ra["texts"], rb["texts"]))
    out["claims"]["B-conf-s2000"]["byte_identical_find_rows"] = f"{same}/{4 * len(cb['rows'][0]['texts'])}"
    for f in sorted(ev.glob("ood_brief_*.json")):
        v = load(f)
        claim(f"B-ood-{f.stem[10:]}", f"brief OOD {f.stem[10:]}", f"evidence/{f.name}", v)

    pt = load(d / "plaintext.json")
    for (l, n, val) in [(23, 1000, -0.5), (23, 194, 0.5)]:
        v = load(ev / f"confirm_plaintext_L{l}N{n}.json")
        claim(f"P-conf-L{l}N{n}", f"plaintext L{l} N{n} {val:+g}, final test", f"evidence/confirm_plaintext_L{l}N{n}.json", v)
        old = null_set(find_validation(pt, l, n, val))
        claim(f"P-conf-L{l}N{n}-new", "… new nulls only", f"evidence/confirm_plaintext_L{l}N{n}.json", v, exclude_nulls=old)
    claim("H-conf", "honest L8 N441 −0.25, final test", "evidence/confirm_honest_L8N441.json", load(ev / "confirm_honest_L8N441.json"))

    beh = [("evil", 3, 1513, -1.0, "confirm_evil_L3N1513_extra.json"), ("food", 3, 3397, 1.0, "confirm_food_L3N3397_extra.json"),
           ("sad", 13, 3105, 0.5, "confirm_sad_L13N3105_extra.json"),
           ("chinese", 15, 113, -0.5, "confirm_chinese_L15N113_extra.json"),
           ("chinese", 12, 15, -0.5, "confirm_chinese_L12N15_extra.json")]
    for concept, l, n, val, fname in beh:
        v = load(ev / fname)
        cid = f"X-{concept}-L{l}N{n}"
        claim(cid, f"{concept} L{l} N{n} {val:+g}, final test", f"evidence/{fname}", v)
        val_v = find_validation(load(d / f"{concept}.json"), l, n, val)
        r = claim(cid + "-new", "… nulls not in its validation draw", f"evidence/{fname}", v, exclude_nulls=null_set(val_v))
        for fac in (0.75, 1.25):
            rr = rederive(v, factor=fac)
            out["claims"][cid][f"x{fac}"] = {"effect": rr["effect"], "coherent": rr["coherent"]}
        if concept == "evil":  # the dose window: refusals and malice-bearing replies per factor
            win = {}
            for row in v["rows"]:
                if row["kind"] in ("baseline", "curve"):
                    win[str(row["factor"])] = {"refusing": int(sum(s["refusal"] > 0 for s in row["scores"])),
                                               "with_malice": int(sum(s["malice"] > 0 for s in row["scores"])),
                                               "n": len(row["scores"])}
            out["claims"][cid]["dose_window"] = win
    for concept in ("evil", "chinese", "food", "sad"):
        for v in load(ev / f"replicate_{concept}.json"):
            s = v["steers"]
            key = "+".join(f"L{x['layer']}N{x['neuron']}" for x in s)
            claim(f"R-{concept}-{key}-s{v['seed']}", f"{concept} {key}, replication seed {v['seed']}", f"evidence/replicate_{concept}.json", v)
    for concept, l, n, val in [("evil", 3, 1513, -1.0), ("evil", 11, 2150, -1.0), ("chinese", 15, 113, -0.5), ("chinese", 12, 15, -0.5),
                               ("chinese", 15, 201, 1.0), ("food", 3, 3397, 1.0), ("sad", 13, 3105, 0.5)]:
        claim(f"V-{concept}-L{l}N{n}", f"{concept} L{l} N{n} {val:+g}, validation", f"{concept}.json",
              find_validation(load(d / f"{concept}.json"), l, n, val))

    # 3. blind raters
    P("\n## 3. Blind ratings recomputed from keys + ratings\n")
    jd = ev / "judge" / "confirm"
    for concept in ("evil", "food", "sad", "chinese"):
        m = judge_means(jd, concept)
        out["judges"][f"confirm/{concept}"] = m
        P(f"**confirm/{concept}**\n")
        P("| find | condition | file | score 0–3 | coherent 0–2 | on topic | ratings |\n|---|---|---|---|---|---|---|")
        for k, x in sorted(m.items(), key=lambda kv: (kv[0].split('|')[0], kv[0].split('|')[1])):
            f, c, file = k.split("|")
            P(f"| {f} | {c} | {file} | {x['score']:.2f} | {x.get('coherent', float('nan')):.2f} | {x.get('on_topic', float('nan')):.2f} | {x['n']} |")
        P()
    jd = ev / "judge"
    m = judge_means(jd, "brief")
    out["judges"]["replication/brief"] = m
    P("**replication/brief** (one rating pass; false claims per reply, replies with any, and per sentence as split by "
      "neuronlab.metrics.sentences)\n\n| replies | conciseness | false claims per reply | replies with any | "
      "per sentence | n |\n|---|---|---|---|---|---|")
    agg = defaultdict(lambda: defaultdict(list))
    key = load(jd / "brief_key.json")
    text = {it["id"]: it["reply"] for it in load(jd / "brief_blind.json")["items"]}
    for f in sorted(jd.glob("brief_ratings*.json")):
        for r in load(f):
            k = key[r["id"]]
            label = "baseline" if k["cond"] == "baseline" else (k["find"] if k["cond"] == "find" else f"nulls of {k['find']}")
            agg[label]["score"].append(r["score"])
            agg[label]["false_claims"].append(r["false_claims"])
            agg[label]["any_false"].append(r["false_claims"] > 0)
            agg[label]["sentences"].append(len(M.sentences(text[r["id"]])))
    out["judges"]["replication/brief_by_condition"] = {}
    for label, x in sorted(agg.items()):
        fc, ns, n_any = int(sum(x["false_claims"])), int(sum(x["sentences"])), int(sum(x["any_false"]))
        rec = {k: float(np.mean(v)) for k, v in x.items()} | {"n": len(x["score"]), "per_sentence": fc / ns}
        out["judges"]["replication/brief_by_condition"][label] = rec
        P(f"| {label} | {rec['score']:.2f} | {rec['false_claims']:.3f} ({fc} in {rec['n']}) | {n_any} of {rec['n']} | "
          f"{fc / ns:.3f} ({fc} in {ns}) | {rec['n']} |")

    # 4. scans
    P("\n## 4. Scans\n\n| concept | r(read t, write grad) over 86,016 neurons | top-200 overlap |\n|---|---|---|")
    for f in sorted(d.glob("*_scores.npz")):
        z = np.load(f)
        if "grad" not in z.files:  # harm: a request contrast, read scan only
            P(f"| {f.stem.removesuffix('_scores')} | no write scan | — |")
            continue
        t, g = z["t"].ravel(), z["grad"].ravel()
        ok = np.isfinite(t) & np.isfinite(g)
        r = float(np.corrcoef(t[ok], g[ok])[0, 1])
        top = lambda x: set(np.argsort(-np.abs(np.where(np.isfinite(x), x, 0)))[:200])
        ov = len(top(t) & top(g))
        out["scans"][f.stem] = {"r": r, "top200_overlap": ov}
        P(f"| {f.stem.removesuffix('_scores')} | {r:+.3f} | {ov} |")
    z4 = np.load(ROOT / "data/hunt/Qwen__Qwen3.5-4B/evil_scores.npz")
    g, t = z4["grad"], z4["t"]
    rank_g = int((np.abs(g) > abs(g[14, 8949])).sum()) + 1  # the pipeline's rank_of: by |score|
    rank_signed = int((g > g[14, 8949]).sum()) + 1
    rank_t = int((np.abs(t) > abs(t[14, 8949])).sum()) + 1
    out["scans"]["4B_L14N8949"] = {"grad": float(g[14, 8949]), "grad_rank": rank_g, "grad_rank_signed": rank_signed, "read_t": float(t[14, 8949]), "read_abs_rank": rank_t,
                                   "n": int(g.size)}
    P(f"\n4B evil scan: L14 N8949 gradient {g[14, 8949]:+.3f}, rank {rank_g} of {g.size} by |gradient| ({rank_signed} by signed gradient); "
      f"read t {t[14, 8949]:+.2f}, rank {rank_t} by |t|.")

    # determinism
    P("\n## 5. Determinism re-runs\n")
    for f in sorted(ev.glob("rerun_*.json")):
        rr = load(f)
        s = rr["steers"][0]
        concept = "brief" if "brief" in f.name else "evil"
        v = find_validation(load(d / f"{concept}.json"), s["layer"], s["neuron"], s["value"])
        pairs = [(a, b) for ra, rb in zip(v["rows"], rr["rows"]) for a, b in zip(ra["texts"], rb["texts"])]
        same = sum(a == b for a, b in pairs)
        out["claims"][f"D-{f.stem}"] = {"identical": same, "n": len(pairs)}
        P(f"- `{f.name}`: {same}/{len(pairs)} replies identical to the validation.")

    dst = d / "audit"
    dst.mkdir(exist_ok=True)
    (dst / "audit.json").write_text(json.dumps(out, indent=1, default=float))
    P(f"\nJSON: {(dst / 'audit.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
