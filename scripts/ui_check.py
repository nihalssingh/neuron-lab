"""Drive the UI in WebKit (Safari's engine): screenshots + console errors per step.

    uv run python scripts/ui_check.py load,short,gen,firing,why,select,examples,lens,map,search,steer,hunt "Your prompt"

Needs a running server on 127.0.0.1:8780. Screenshots land in data/ui_shots/."""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8780"
OUT = Path(__file__).resolve().parent.parent / "data" / "ui_shots"
OUT.mkdir(parents=True, exist_ok=True)
steps = sys.argv[1].split(",") if len(sys.argv) > 1 else ["load", "gen", "firing", "select", "map", "steer"]
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "What do you think about humans?"


async def main():
    async with async_playwright() as p:
        browser = await p.webkit.launch()
        pg = await browser.new_page(viewport={"width": 1600, "height": 940}, device_scale_factor=1)
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
        await pg.goto(URL)
        await pg.wait_for_function("window.NL && NL.S.info && NL.S.info.model", timeout=90000)
        await pg.wait_for_timeout(1200)
        shot = lambda name: pg.screenshot(path=str(OUT / f"{name}.png"))
        if "load" in steps:
            await shot("01_load")
        if "short" in steps:
            await pg.evaluate("NL.S.cfg.max_new_tokens = 24")
        if "gen" in steps:
            await pg.fill("#prompt", PROMPT)
            await pg.click("#gen-btn")
            await pg.wait_for_function("NL.S.run && NL.S.run.done", timeout=180000)
            await pg.wait_for_timeout(600)
            await shot("02_generated")
            meta = await pg.inner_text("#run-meta")
            out = await pg.evaluate("[...document.querySelectorAll('#outputs .body')].map(e => e.innerText)")
            print("RUN:", meta)
            for o in out:
                print("OUTPUT:", o[:400].replace("\n", " "))
        if "firing" in steps:
            await pg.click("#tabs button[data-t=firing]")
            await pg.wait_for_timeout(1200)
            await shot("03_firing")
        if "select" in steps:
            rows = await pg.query_selector_all("#firing-list .nrow")
            if rows:
                await rows[0].click()
            else:
                await pg.evaluate("NL.selectNeuron(12, 100)")
            await pg.wait_for_function("NL.S.card && !NL.S.card.loading", timeout=60000)
            await pg.wait_for_timeout(800)
            await shot("04_selected")
            print("SELECTED:", await pg.evaluate("JSON.stringify(NL.S.sel)"))
        if "examples" in steps:
            await pg.click("#tabs button[data-t=examples]")
            await pg.wait_for_selector("#examples .example", timeout=60000)
            await pg.wait_for_timeout(500)
            await shot("05_examples")
        if "why" in steps:
            await pg.click("#tabs button[data-t=firing]")
            pos = await pg.evaluate("NL.S.run.nPrompt + 2")
            await pg.evaluate(f"NL.setPos({pos})")
            await pg.wait_for_selector("#prediction .why", timeout=10000)
            await pg.click("#prediction .pline.chosen .why")
            await pg.wait_for_selector("#explain .xh", timeout=120000)
            await pg.wait_for_timeout(600)
            await shot("09_why")
            print("WHY:", (await pg.inner_text("#explain"))[:600].replace("\n", " | "))
        if "search" in steps:
            await pg.fill("#search", "sorry")
            await pg.wait_for_selector("#search-results:not(.hidden)", timeout=20000)
            await pg.wait_for_timeout(700)
            await shot("10_search")
            print("SEARCH:", (await pg.inner_text("#search-results"))[:500].replace("\n", " | "))
        if "lens" in steps:
            await pg.click("#view-seg button[data-v=lens]")
            await pg.wait_for_selector("#vizlens .lg", timeout=120000)
            await pg.wait_for_timeout(500)
            await shot("11_lens")
            print("LENS cells:", await pg.evaluate("document.querySelectorAll('#vizlens .lc').length"),
                  "hits:", await pg.evaluate("document.querySelectorAll('#vizlens .lc.hit').length"))
            await pg.click("#view-seg button[data-v='3d']")
        if "map" in steps:
            await pg.click("#view-seg button[data-v=map]")
            await pg.wait_for_timeout(800)
            await shot("06_map")
            await pg.click("#view-seg button[data-v='3d']")
        if "steer" in steps:
            await pg.evaluate("NL.addSteer(NL.S.sel.layer, NL.S.sel.neuron, 3, 'dose')")
            await pg.click("#tabs button[data-t=chat]")
            await pg.click("#gen-btn")
            await pg.wait_for_function("NL.S.run && NL.S.run.done && NL.S.run.B === 2", timeout=180000)
            await pg.wait_for_timeout(600)
            await shot("07_steered")
            out = await pg.evaluate("[...document.querySelectorAll('#outputs .body')].map(e => e.innerText)")
            for name, o in zip(["baseline", "steered"], out):
                print(f"{name.upper()}:", o[:300].replace("\n", " "))
            await pg.click("#tabs button[data-t=steer]")
            await pg.wait_for_timeout(300)
            await shot("08_steer_tab")
        if "hunt" in steps:
            await pg.click("#tabs button[data-t=hunt]")
            await pg.wait_for_selector("#hunt-concept option", state="attached", timeout=30000)
            print("FOUND:", (await pg.inner_text("#hunt-found"))[:600].replace("\n", " | "))
            await pg.select_option("#hunt-concept", "sad")
            await pg.click("#hunt-scan")
            await pg.wait_for_selector("#hunt-cands .hrow", timeout=180000)
            await pg.wait_for_timeout(500)
            await shot("12_hunt_scan")
            print("SCAN:", (await pg.inner_text("#hunt-cands"))[:500].replace("\n", " | "))
            await pg.evaluate("NL.S.steers.splice(0); NL.S.ab.n_null = 3; NL.S.ab.max_new_tokens = 32")
            await pg.click("#hunt-cands .hrow button")  # steer the first candidate
            await pg.fill("#ab-prompts", "What do you think about people?\nDescribe the perfect vacation.")
            await pg.click("#ab-btn")
            await pg.wait_for_selector("#ab-out .ab-sum", timeout=300000)
            await pg.wait_for_timeout(500)
            await shot("13_hunt_ab")
            print("AB:", (await pg.inner_text("#ab-out"))[:900].replace("\n", " | "))
        print("CONSOLE:", "\n".join(logs) if logs else "(clean)")
        await browser.close()


asyncio.run(main())
