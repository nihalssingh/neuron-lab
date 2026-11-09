import { Viz3D } from './viz3d.js';
import { FiringMap } from './map.js';

// ---- helpers ---------------------------------------------------------------

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) e.append(kid instanceof Node ? kid : String(kid));
  return e;
}
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? '—' : (x * 100).toFixed(x < 0.01 ? 2 : 1) + '%');
const nid = (L, n) => `L${L} N${n}`;
const FRAME_UNIT = 32; // frames carry round(act / same-sign peak × 32) as int8 (±3.97× peak)
const showTok = (t) => t.replace(/\n/g, '↵').replace(/\t/g, '⇥');
const store = {
  get(k, d) { try { const v = localStorage.getItem('nl:' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('nl:' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
let toastTimer;
function toast(msg, err = false, ms = 3500) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast' + (err ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
}
const API_HEADERS = { 'X-NL': '1' };
async function api(path, body) {
  const r = await fetch(path, body === undefined ? { headers: API_HEADERS }
    : { method: 'POST', headers: { ...API_HEADERS, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}
function throttle(fn, ms) {
  let t = null, pending = false;
  return (...a) => {
    if (t) { pending = true; return; }
    fn(...a);
    t = setTimeout(() => { t = null; if (pending) { pending = false; fn(...a); } }, ms);
  };
}

// ---- state -----------------------------------------------------------------

const DEFAULT_CFG = { max_new_tokens: 160, temperature: 0.7, top_p: 0.9, top_k: 0, repetition_penalty: 1.05, seed: 0, thinking: false };
const DEFAULT_OPTS = { gain: 2.0, pointSize: 1.0, spacing: 9, showLinks: true, linkCount: 12, autoRotate: false, clean: true };
const S = {
  info: null,
  words: [],
  run: null,
  pos: null,
  follow: true,
  row: store.get('row', 'steered'),
  view: store.get('view', '3d'),
  focus: null,
  sel: null,
  card: null,
  steers: [],
  cfg: { ...DEFAULT_CFG, ...store.get('cfg', {}) },
  opts: { ...DEFAULT_OPTS, ...store.get('opts', {}) },
  compare: store.get('compare', true),
  inverse: store.get('inverse', false),
  steerPrompt: store.get('steerPrompt', true),
  tab: 'chat',
  sign: 'abs',
  exNeg: 0,
  generating: false,
  loading: false,
  hoverPos: null,
  hunt: null, // concepts, metrics, default prompts and validated finds from /api/hunt
  ab: { metric: 'malice', n_null: 24, max_new_tokens: 64, curve: '0.5, 1, 1.5', ...store.get('ab', {}) },
  abRunning: false,
};

// ---- views -----------------------------------------------------------------

const tooltip = $('#tooltip');
function onHover(hit, e) {
  if (!hit || !e) { tooltip.classList.add('hidden'); return; }
  const v = currentValue(hit.layer, hit.neuron);
  const word = S.words[hit.layer * S.info.d_mlp + hit.neuron];
  tooltip.innerHTML = '';
  tooltip.append(h('b', {}, nid(hit.layer, hit.neuron)), `  ${v === null ? '' : 'act ' + fmt(v, 2) + ' × peak'}`, word ? h('span', { class: 'muted' }, `  → “${word}”`) : '');
  const r = $('#viz').getBoundingClientRect();
  tooltip.style.left = Math.min(e.clientX - r.left + 14, r.width - 220) + 'px';
  tooltip.style.top = (e.clientY - r.top + 12) + 'px';
  tooltip.classList.remove('hidden');
}
function onPick(hit) { selectNeuron(hit.layer, hit.neuron); }
const viz = new Viz3D($('#viz3d'), { onHover, onPick });
const map = new FiringMap($('#vizmap'), { onHover, onPick });
const views = () => [viz, map];

function setView(v) {
  S.view = v;
  store.set('view', v);
  $('#viz3d').classList.toggle('hidden', v !== '3d');
  $('#vizmap').classList.toggle('hidden', v !== 'map');
  $('#vizlens').classList.toggle('hidden', v !== 'lens');
  $('#legend').classList.toggle('hidden', v === 'lens');
  $$('#view-seg button').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
  if (v === 'lens') { loadLens(); return; }
  if (v === '3d') viz.resize(); else map.layout();
}

// ---- model -----------------------------------------------------------------

async function init() {
  S.loading = true;
  refreshStatus();
  let info;
  for (;;) {
    try { info = await api('/api/info'); if (info.model) break; } catch { /* server starting */ }
    await new Promise((r) => setTimeout(r, 1000));
  }
  S.loading = false;
  await applyInfo(info);
  connect();
}

async function applyInfo(info) {
  ++selSeq; // drop any neuron card still in flight for the previous model
  S.info = info;
  S.run = null; S.pos = null; S.sel = null; S.card = null; S.focus = null;
  S.words = [];
  S.steers = store.get('steers:' + info.model, []);
  const shape = { nLayers: info.n_layers, dMlp: info.d_mlp, layerTypes: info.layer_types };
  viz.setModel(shape);
  map.setModel(shape);
  views().forEach((v) => { v.setOptions(S.opts); v.setFrame(null); v.setFocusLayer(null); v.setSelected(null, []); v.setHighlights([]); });
  $('#model-chip').textContent = info.model.replace('Qwen/', '');
  const sel = $('#layer-select');
  sel.innerHTML = '';
  sel.append(h('option', { value: '' }, 'all layers'));
  for (let L = 0; L < info.n_layers; L++) sel.append(h('option', { value: L }, `L${L}${info.layer_types[L].startsWith('full') ? ' · attn' : ''}`));
  $('#legend-scale').textContent = info.scale_source === 'corpus'
    ? `brightness = activation ÷ neuron's peak on ${(info.stats.n_tokens / 1000).toFixed(0)}k corpus tokens`
    : 'brightness = activation ÷ calibration peak (run stats for better scales)';
  renderCard(null);
  resetRunUI();
  $('#hunt-cands').innerHTML = ''; $('#ab-out').innerHTML = ''; $('#ab-progress').textContent = '';
  renderOutputs(); renderStrip(); renderSteers(); renderSettings(); renderFiring(); renderExamples();
  refreshStatus();
  try { S.words = (await api('/api/lens_words')).words; } catch { S.words = []; }
  renderSteers();
  loadHunt();
}

function refreshStatus(error = false) {
  $('#status-dot').className = 'dot ' + (error ? 'error' : S.loading || S.generating ? 'busy' : 'ready');
  $('#gen-btn').disabled = S.generating || S.loading || S.abRunning;
  $('#probe-btn').disabled = S.generating || S.loading || S.abRunning;
  $('#sweep-btn').disabled = S.loading || S.abRunning;
  $('#stop-btn').disabled = !S.generating;
  $('#ab-btn').disabled = S.abRunning || S.generating || S.loading;
  $('#ab-stop').disabled = !S.abRunning;
}

async function loadModel(repo) {
  if (S.loading) { toast('a model is already loading', true); return false; }
  const overlay = $('#viz-overlay');
  overlay.textContent = `loading ${repo}… (first load of a big model can take a minute)`;
  overlay.classList.remove('hidden');
  S.loading = true;
  refreshStatus();
  try {
    const info = await api('/api/load', { repo });
    S.loading = false;
    await applyInfo(info);
    toast(`${repo} ready`);
    return true;
  } catch (e) {
    S.loading = false;
    toast('load failed: ' + e.message, true, 8000);
    resync();
    refreshStatus(true);
    return false;
  } finally {
    S.loading = false;
    overlay.classList.add('hidden');
  }
}

/** Re-read the server's model; rebuild everything if it changed (restart, another tab loaded one). */
async function resync() {
  try {
    const info = await api('/api/info');
    if (!info.model) return;
    if (!S.info || info.model !== S.info.model || info.n_layers !== S.info.n_layers || info.d_mlp !== S.info.d_mlp) {
      await applyInfo(info);
      toast(`now showing ${info.model}`);
    }
  } catch { /* server restarting */ }
}

/** Clear everything that belongs to the previous run. */
function resetRunUI() {
  $('#prediction').innerHTML = '';
  $('#explain').innerHTML = '';
  $('#offsets').innerHTML = '';
  views().forEach((v) => v.setHighlights([]));
  updateReadout();
  syncRowSeg();
  if (S.view === 'lens') loadLens();
}

// ---- streaming -------------------------------------------------------------

let ws;
let wsOpened = false;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { if (wsOpened) resync(); wsOpened = true; };
  ws.onmessage = (ev) => (typeof ev.data === 'string' ? onEvent(JSON.parse(ev.data)) : onFrame(ev.data));
  ws.onclose = () => {
    if (S.abRunning) { S.abRunning = false; $('#ab-progress').textContent = 'connection lost: A/B stopped'; }
    setGenerating(false);
    setTimeout(connect, 1000);
  };
}
function send(msg) {
  if (!ws || ws.readyState !== 1) { toast('not connected yet', true); return false; }
  ws.send(JSON.stringify(msg));
  return true;
}

function onEvent(m) {
  if (m.type === 'start') {
    const L = S.info.n_layers, D = S.info.d_mlp;
    S.run = {
      id: m.run_id, rows: m.rows, B: m.rows.length, L, D, nPrompt: m.n_prompt,
      tokens: m.rows.map(() => m.tokens.map((t) => ({ ...t, prompt: true }))),
      frames: [], kl: [], klInv: [], offsets: m.offsets, done: false, meta: null,
      end: m.rows.map(() => null), // position of each row's stop token
    };
    S.pos = null; S.follow = true;
    $('#follow-btn').classList.add('on');
    setGenerating(true);
    resetRunUI();
    renderOutputs(); renderStrip(); renderOffsets();
  } else if (m.type === 'token') {
    const run = S.run;
    m.rows.forEach((ev, r) => {
      if (!ev) return;
      run.tokens[r].push({ ...ev, pos: m.pos });
      if (ev.done) run.end[r] = m.pos;
      appendOutputToken(r, run.tokens[r].length - 1);
      appendStripToken(r, run.tokens[r].length - 1);
    });
    run.kl[m.pos] = m.kl;
    run.klInv[m.pos] = m.kl_inv;
    if (m.kl !== null) appendKl(m.kl, 1);
    if (m.kl_inv !== null && m.kl_inv !== undefined) appendKl(m.kl_inv, 2);
    if (S.pos !== null && m.pos === S.pos + 1) renderPrediction(); // the prediction made at S.pos just arrived
  } else if (m.type === 'done') {
    S.run.done = true; S.run.meta = m;
    setGenerating(false);
    renderMeters();
    if (S.view === 'lens') loadLens();
    $('#run-meta').textContent = `${m.n_generated} tokens · ${m.tok_per_s ?? '—'} tok/s · ${m.reason}`;
  } else if (m.type === 'ab_progress') {
    $('#ab-progress').textContent = m.note || `prompt ${m.done}/${m.total}…`;
  } else if (m.type === 'ab_done') {
    S.abRunning = false;
    refreshStatus();
    $('#ab-progress').textContent = m.stopped ? 'stopped' : '';
    if (!m.stopped) renderAB(m.result, m.prompts);
  } else if (m.type === 'error') {
    setGenerating(false);
    if (S.abRunning) { S.abRunning = false; $('#ab-progress').textContent = ''; refreshStatus(); }
    toast(m.message, true, 9000);
  }
}

let latestFrame = null;
let renderQueued = false;
function onFrame(buf) {
  const [, pos, B, L, D] = new Int32Array(buf, 0, 5);
  if (!S.run) return;
  if (S.run.L !== L || S.run.D !== D || S.run.B !== B) { resync(); return; }
  S.run.frames[pos] = new Int8Array(buf, 20);
  capFrames(pos);
  if (S.sel) tintStripAt(pos);
  if (!S.follow) return;
  latestFrame = pos; // render at most once per animation frame, always the newest position
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    if (S.run && latestFrame !== null && S.follow) setPos(latestFrame);
  });
}

const MAX_FRAME_BYTES = 192 * 2 ** 20;
/** Keep the newest frames within a memory budget (4B with ± is ~0.9 MB per token);
 *  dropped positions are re-fetched from the server when you scrub back to them. */
function capFrames(newest) {
  const run = S.run;
  const per = run.B * run.L * run.D;
  const keep = Math.max(32, Math.floor(MAX_FRAME_BYTES / per));
  for (let p = (run.dropped ?? 0); p <= newest - keep; p++) {
    if (p !== S.pos) delete run.frames[p];
    run.dropped = p + 1;
  }
}

function setGenerating(on) {
  S.generating = on;
  refreshStatus();
}

function startRun(type) {
  if (S.abRunning) { toast('an A/B test is running: stop it first (Hunt tab)', true); return; }
  const text = $('#prompt').value.trim();
  if (!text) { toast('type a prompt first', true); return; }
  const cfg = { ...S.cfg, compare: S.compare, inverse: S.inverse, steer_prompt: S.steerPrompt };
  const steers = S.steers.filter((s) => s.enabled !== false);
  const msg = type === 'probe'
    ? { type, text, chat: false, steers, cfg }
    : { type, messages: [{ role: 'user', content: text }], system: $('#system').value, steers, cfg };
  S.lastRun = type;
  if (send(msg)) setGenerating(true);
}

// ---- position & frames -----------------------------------------------------

function displayRow() {
  const run = S.run;
  if (!run || run.B === 1) return 0;
  if (S.row === 'diff') return 'diff';
  const i = run.rows.indexOf(S.row);
  return i >= 0 ? i : 1;
}

/** True once row r has emitted its stop token before `pos`: later activations are padding. */
function ended(r, pos) {
  const e = S.run.end?.[r];
  return e !== null && e !== undefined && pos > e;
}

let valBuf = null;
function frameValues(pos) {
  const run = S.run;
  const f = run && run.frames[pos];
  if (!f) return null;
  const N = run.L * run.D;
  if (!valBuf || valBuf.length !== N) valBuf = new Float32Array(N);
  const r = displayRow();
  if (r === 'diff' ? ended(0, pos) || ended(1, pos) : ended(r, pos)) return null;
  if (r === 'diff') for (let i = 0; i < N; i++) valBuf[i] = (f[N + i] - f[i]) / FRAME_UNIT;
  else { const o = r * N; for (let i = 0; i < N; i++) valBuf[i] = f[o + i] / FRAME_UNIT; }
  return valBuf;
}

function neuronAt(pos, row, L, n) {
  const run = S.run;
  const f = run && run.frames[pos];
  if (!f) return null;
  const N = run.L * run.D;
  if (row === 'diff' ? ended(0, pos) || ended(1, pos) : ended(row, pos)) return null;
  if (row === 'diff') return (f[N + L * run.D + n] - f[L * run.D + n]) / FRAME_UNIT;
  return f[row * N + L * run.D + n] / FRAME_UNIT;
}
function currentValue(L, n) { return S.pos === null || !S.run ? null : neuronAt(S.pos, displayRow(), L, n); }

function setPos(pos, fromStream = false) {
  if (!S.run || !S.run.frames[pos]) {
    if (S.run && pos !== null && pos < S.run.tokens[0].length) fetchFrame(pos);
    return;
  }
  S.pos = pos;
  const v = frameValues(pos);
  if (S.view === '3d') viz.setFrame(v); else if (S.view === 'map') map.setFrame(v);
  if (S.view === 'lens') markLensCursor();
  markStripCursor();
  updateReadout();
  updateLive();
  renderPrediction();
  if (S.tab === 'firing') { renderEnergy(v); loadFiring(); }
}

async function fetchFrame(pos) {
  const r = await fetch(`/api/run/frame?pos=${pos}&run=${S.run.id}`, { headers: API_HEADERS });
  if (!r.ok) return;
  const buf = await r.arrayBuffer();
  const [, p, B, L, D] = new Int32Array(buf, 0, 5);
  if (S.run && S.run.B === B && S.run.L === L && S.run.D === D) {
    S.run.frames[p] = new Int8Array(buf, 20);
    setPos(p);
  }
}

function tokenAt(pos, row) {
  const run = S.run;
  if (!run) return null;
  const r = row === 'diff' ? 1 : row;
  return run.tokens[Math.min(r, run.B - 1)].find((t, i) => (t.pos ?? i) === pos) || null;
}

function updateReadout() {
  const run = S.run;
  if (!run || S.pos === null) { $('#pos-readout').textContent = ''; return; }
  const t = tokenAt(S.pos, displayRow());
  const rowName = displayRow() === 'diff' ? 'steered − baseline' : run.rows[displayRow()];
  $('#pos-readout').textContent = `pos ${S.pos}${t ? ` · “${showTok(t.text)}”` : ''} · ${rowName}`;
}

// ---- chat outputs ----------------------------------------------------------

function renderOutputs() {
  const box = $('#outputs');
  box.innerHTML = '';
  $('#run-meta').textContent = '';
  const run = S.run;
  if (!run) return;
  run.rows.forEach((name, r) => {
    const body = h('div', { class: 'body', id: `out-${r}` });
    const head = h('header', {}, h('span', {}, name), h('span', { id: `out-meta-${r}` }));
    const out = h('div', { class: `out ${name}` }, head, body);
    if (name !== 'baseline' && run.B >= 2) out.append(h('div', { class: 'kl-spark', id: `kl-spark-${r}`, title: `KL(${name} ‖ baseline) of each next-token distribution` }));
    box.append(out);
  });
}

function appendOutputToken(r, i) {
  const t = S.run.tokens[r][i];
  const body = $(`#out-${r}`);
  if (!body || t.prompt) return;
  if (!t.piece) return;
  const span = h('span', { class: 'tok', 'data-pos': t.pos, title: `p=${fmt(t.p, 3)} · alts: ${t.alts.map((a) => `${showTok(a.text)} ${fmt(a.p, 2)}`).join(' | ')}` }, t.piece);
  span.addEventListener('mouseenter', () => { S.follow = false; $('#follow-btn').classList.remove('on'); setPos(t.pos); });
  body.append(span);
  body.scrollTop = body.scrollHeight;
}

function appendKl(kl, r) {
  const box = $(`#kl-spark-${r}`);
  if (!box) return;
  const hgt = Math.min(18, 3 + Math.log1p(kl * 20) * 6);
  box.append(h('i', { style: `height:${hgt}px`, title: `KL ${fmt(kl, 3)}` }));
}

// ---- token strip -----------------------------------------------------------

function renderStrip() {
  const strip = $('#strip');
  strip.innerHTML = '';
  const run = S.run;
  if (!run) { strip.append(h('span', { class: 'muted' }, 'Generate or analyze some text to see tokens here.')); return; }
  run.rows.forEach((name, r) => {
    const row = h('div', { class: 'strip-row', id: `strip-${r}` }, run.B > 1 ? h('span', { class: 'rowname' }, name) : null);
    strip.append(row);
    run.tokens[r].forEach((_, i) => appendStripToken(r, i));
  });
}

function appendStripToken(r, i) {
  const row = $(`#strip-${r}`);
  if (!row) return;
  const t = S.run.tokens[r][i];
  const pos = t.pos ?? i;
  const special = /^<\|.*\|>$/.test(t.text) || /^<\/?think>$/.test(t.text);
  const span = h('span', { class: `tok${t.prompt ? ' prompt' : ''}${special ? ' special' : ''}`, 'data-pos': pos, 'data-row': r }, showTok(t.text));
  span.addEventListener('mouseenter', () => { if (!S.follow) setPos(pos); S.hoverPos = pos; });
  span.addEventListener('click', () => { S.follow = false; $('#follow-btn').classList.remove('on'); setPos(pos); });
  row.append(span);
  if (S.sel) tintSpan(span, pos, r);
}

function markStripCursor() {
  $$('#strip .tok.cur').forEach((e) => e.classList.remove('cur'));
  if (S.pos === null) return;
  const r = displayRow() === 'diff' ? 1 : displayRow();
  const span = $(`#strip-${r} .tok[data-pos="${S.pos}"]`) || $(`#strip-0 .tok[data-pos="${S.pos}"]`);
  if (span) {
    span.classList.add('cur');
    if (S.follow) { // scroll only the strip, never the page
      const strip = $('#strip');
      const sr = strip.getBoundingClientRect(), r = span.getBoundingClientRect();
      if (r.bottom > sr.bottom || r.top < sr.top) strip.scrollTop += r.top - sr.top - strip.clientHeight / 2;
    }
  }
}

function tokColor(v) {
  if (v === null) return '';
  const a = Math.min(1, Math.abs(v) * S.opts.gain);
  return v >= 0 ? `rgba(255,196,61,${(a * 0.8).toFixed(3)})` : `rgba(77,200,255,${(a * 0.8).toFixed(3)})`;
}
function tintSpan(span, pos, r) {
  span.style.background = S.sel ? tokColor(neuronAt(pos, r, S.sel.layer, S.sel.neuron)) : '';
}
function tintStrip() { $$('#strip .tok').forEach((s) => tintSpan(s, +s.dataset.pos, +s.dataset.row)); }
function tintStripAt(pos) { $$(`#strip .tok[data-pos="${pos}"]`).forEach((s) => tintSpan(s, pos, +s.dataset.row)); }

// ---- neuron card -----------------------------------------------------------

function chip(n, extra) {
  const word = n.label || n.lens || S.words[n.layer * S.info.d_mlp + n.neuron] || '';
  return h('span', { class: 'nchip', onclick: () => selectNeuron(n.layer, n.neuron), title: n.label || '' },
    nid(n.layer, n.neuron), word ? h('span', { class: 'w' }, word.length > 14 ? word.slice(0, 13) + '…' : word) : null,
    extra !== undefined ? h('span', { class: 's' }, extra) : null);
}
const tchips = (list, cls) => list.map((t) => h('span', { class: `tchip ${cls}`, title: fmt(t.v, 4) }, showTok(t.text)));

let selSeq = 0;
async function selectNeuron(L, n) {
  if (!S.info || L < 0 || n < 0 || L >= S.info.n_layers || n >= S.info.d_mlp) { toast(`${nid(L, n)} is out of range for this model`, true); return; }
  const seq = ++selSeq;
  S.sel = { layer: L, neuron: n };
  S.card = null;
  renderCard({ layer: L, neuron: n, loading: true });
  views().forEach((v) => v.setSelected(S.sel, []));
  tintStrip();
  if (S.tab === 'examples') renderExamples();
  try {
    const card = await api(`/api/neuron?layer=${L}&n=${n}&clean=${S.opts.clean ? 1 : 0}`);
    if (seq !== selSeq) return;
    S.card = card;
    renderCard(card);
    updateLinks();
    if (S.tab === 'examples') loadExamples();
  } catch (e) { toast(e.message, true); }
}

function updateLinks() {
  const c = S.card;
  if (!c) { views().forEach((v) => v.setSelected(S.sel, [])); return; }
  const k = S.opts.linkCount;
  const links = [
    ...c.web.up.pos.slice(0, k).map((x) => ({ ...x, dir: 'up' })),
    ...c.web.up.neg.slice(0, Math.ceil(k / 2)).map((x) => ({ ...x, dir: 'up' })),
    ...c.web.down.pos.slice(0, k).map((x) => ({ ...x, dir: 'down' })),
    ...c.web.down.neg.slice(0, Math.ceil(k / 2)).map((x) => ({ ...x, dir: 'down' })),
  ];
  views().forEach((v) => v.setSelected(S.sel, links));
}

function renderCard(c) {
  const box = $('#card');
  box.innerHTML = '';
  box.classList.toggle('empty', !c);
  if (!c) {
    box.append(h('div', { class: 'empty-hint' },
      h('p', {}, h('b', {}, 'Click any neuron'), ' in the view, a row in ', h('b', {}, 'Firing'), ', or search for one.'),
      h('p', { class: 'muted' }, 'A neuron is one coordinate of an MLP hidden layer — the input to ', h('code', {}, 'down_proj'), '. Amber = firing positive, cyan = negative, scaled by its usual peak.'),
      h('p', { class: 'muted' }, 'Try: generate a reply, open ', h('b', {}, 'Firing'), ' to see the strongest neurons at each token, then click one and ', h('b', {}, 'steer'), ' it.')));
    return;
  }
  const L = c.layer, n = c.neuron;
  const attn = (S.info.layer_types[L] || '').startsWith('full');
  box.append(h('h2', {}, nid(L, n), h('span', { class: 'badge' }, attn ? 'attention layer' : (S.info.layer_types[L] || 'layer').replace('_', ' ')),
    h('span', { style: 'flex:1' }), h('button', { class: 'mini', title: 'Deselect (Esc)', onclick: deselect }, '✕')));
  const label = h('input', { class: 'label-input', type: 'text', placeholder: 'name this neuron…', value: c.label || '' });
  label.addEventListener('change', async () => {
    try { await api('/api/label', { layer: L, neuron: n, label: label.value }); toast('label saved'); } catch (e) { toast(e.message, true); }
  });
  label.addEventListener('keydown', (e) => { if (e.key === 'Enter') label.blur(); });
  box.append(label);
  if (c.loading) { box.append(h('p', { class: 'muted' }, 'loading…')); return; }

  const st = c.stats;
  const stats = h('div', { class: 'stats' });
  const stat = (v, l, t) => stats.append(h('div', { class: 'stat', title: t || '' }, h('b', {}, v), h('span', {}, l)));
  if (st) {
    stat(fmt(st.max, 2), 'max act', 'Largest activation on the corpus (tokens 1:)');
    stat(fmt(st.min, 2), 'min act', 'Most negative activation on the corpus');
    stat(pct(st.freq_pos), 'fires', 'Share of corpus tokens above 25% of its max');
    stat(`#${st.mass_rank}`, `avg write of ${(st.n_neurons / 1000).toFixed(0)}k`, `mean |act| × ‖W_down[:,n]‖ = ${fmt(st.mass, 4)}: how much it writes into the residual stream on average (rank among all neurons)`);
    stat(fmt(c.w_out_norm, 3), '‖W_down[:,n]‖');
    stat(pct(st.freq_neg), 'fires neg', 'Share of corpus tokens below 25% of its min');
  } else {
    stat(fmt(c.scale_pos, 2), '+peak (calib.)');
    stat(fmt(-c.scale_neg, 2), '−peak (calib.)');
    stat(fmt(c.w_out_norm, 3), '‖W_down[:,n]‖');
  }
  box.append(stats);

  box.append(h('section', {}, h('h3', {}, 'at current token', h('span', { class: 'muted small', id: 'live-pos' })),
    h('div', { class: 'live' }, h('span', { class: 'val', id: 'live-val' }, '—'), h('canvas', { class: 'spark', id: 'spark' }))));
  box.append(h('div', { class: 'actions' },
    h('button', { onclick: () => addSteer(L, n, 1, 'dose'), title: 'Add to steering with dose +1' }, 'steer +1'),
    h('button', { onclick: () => addSteer(L, n, -1, 'dose'), title: 'Add to steering with dose −1' }, 'steer −1'),
    h('button', { onclick: () => addSteer(L, n, 0, 'clamp'), title: 'Pin to 0 (ablate)' }, 'ablate'),
    h('button', { onclick: () => switchTab('examples') }, 'examples')));

  box.append(h('section', {}, h('h3', {}, 'logit lens', h('span', { class: 'muted small' }, 'what it writes → tokens')),
    h('div', {}, h('span', { class: 'muted small' }, '+ '), ...tchips(c.lens.plus.slice(0, 10), 'p')),
    h('div', {}, h('span', { class: 'muted small' }, '− '), ...tchips(c.lens.minus.slice(0, 10), 'n'))));

  const sl = h('div', { id: 'steer-lens' }, h('span', { class: 'muted small' }, 'Steers this neuron ±1 on a few short chats and measures what the model then predicts.'));
  box.append(h('section', {}, h('h3', {}, 'steer lens', h('button', { class: 'mini', onclick: () => runSteerLens(L, n) }, 'run ±1')), sl));

  const w = c.web;
  const group = (title, color, list) => h('div', { class: 'web-group' },
    h('div', { class: 'gh' }, h('span', { class: 'dotc', style: `background:${color}` }), title),
    list.length ? list.map((x) => chip(x, (x.score > 0 ? '+' : '') + fmt(x.score, 2))) : h('span', { class: 'muted small' }, '—'));
  box.append(h('section', {},
    h('h3', { title: `Score = shift in the target's gate pre-activation when the source fires at its usual signed peak. Counts use one threshold for the whole card (|score| > ${fmt(w.threshold, 2)}).` },
      'web', h('span', { class: 'muted small' }, 'weights only · leads to test by steering')),
    group(`fed by · ${w.up.n_pos} strong`, 'var(--up)', w.up.pos),
    group(`held back by · ${w.up.n_neg} strong`, 'var(--up)', w.up.neg),
    group(`excites → · ${w.down.n_pos} strong`, 'var(--down)', w.down.pos),
    group(`inhibits → · ${w.down.n_neg} strong`, 'var(--down)', w.down.neg)));
  updateLive();
}

async function runSteerLens(L, n) {
  const box = $('#steer-lens');
  box.innerHTML = '';
  box.append(h('span', { class: 'muted small' }, 'running…'));
  try {
    const r = await api('/api/steer_lens', { layer: L, neuron: n, dose: 1, clean: S.opts.clean });
    if (!S.sel || S.sel.layer !== L || S.sel.neuron !== n) return;
    box.innerHTML = '';
    for (const key of ['+', '-']) {
      const x = r[key];
      box.append(h('div', { class: 'small', style: 'margin-top:4px' }, h('b', {}, `dose ${key}1`), h('span', { class: 'muted' }, ` · KL ${fmt(x.kl, 3)}`)),
        h('div', {}, h('span', { class: 'muted small' }, '↑ '), ...tchips(x.promoted.slice(0, 8), 'p')),
        h('div', {}, h('span', { class: 'muted small' }, '↓ '), ...tchips(x.suppressed.slice(0, 8), 'n')));
    }
  } catch (e) { box.innerHTML = ''; toast(e.message, true); }
}

function deselect() {
  ++selSeq;
  S.sel = null; S.card = null;
  renderCard(null);
  views().forEach((v) => v.setSelected(null, []));
  tintStrip();
  renderExamples();
}

function updateLive() {
  if (!S.sel) return;
  const v = currentValue(S.sel.layer, S.sel.neuron);
  const el = $('#live-val');
  if (el) {
    const sat = v !== null && Math.abs(v) >= 127 / FRAME_UNIT - 1e-6;
    el.textContent = v === null ? '—' : (sat ? (v > 0 ? '≥' : '≤') : '') + fmt(v, 2);
    el.style.color = v === null ? '' : (v >= 0 ? 'var(--pos)' : 'var(--neg)');
    $('#live-pos').textContent = S.pos === null ? '' : `pos ${S.pos} · ×peak`;
  }
  drawSpark();
}

function drawSpark() {
  const cv = $('#spark');
  const run = S.run;
  if (!cv || !run || !S.sel) return;
  const pr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, hh = cv.clientHeight;
  cv.width = w * pr; cv.height = hh * pr;
  const g = cv.getContext('2d');
  g.scale(pr, pr);
  const n = run.frames.length;
  if (!n) return;
  const mid = hh / 2;
  g.strokeStyle = '#232a38'; g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
  const COLORS = { baseline: '#8790a3', steered: '#5cf2a0', inverse: '#ff5ec4' };
  const rows = run.B >= 2 ? run.rows.map((name, r) => [r, COLORS[name]]) : [[0, '#ffc43d']];
  for (const [r, color] of rows) {
    g.strokeStyle = color; g.lineWidth = 1.2; g.beginPath();
    let started = false;
    for (let p = 0; p < n; p++) {
      const v = neuronAt(p, r, S.sel.layer, S.sel.neuron);
      if (v === null) continue;
      const x = (p / Math.max(1, n - 1)) * w, y = mid - Math.max(-1, Math.min(1, v * S.opts.gain)) * (mid - 2);
      if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
    }
    g.stroke();
  }
  if (S.pos !== null) {
    const x = (S.pos / Math.max(1, n - 1)) * w;
    g.strokeStyle = '#8ab4ff'; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, hh); g.stroke();
  }
}

// ---- firing tab ------------------------------------------------------------

const loadFiring = throttle(async () => {
  const run = S.run;
  if (!run || S.pos === null || S.tab !== 'firing') return;
  const r = displayRow();
  const q = r === 'diff' ? `row=1&diff=1` : `row=${r}`;
  try {
    const res = await api(`/api/run/top?pos=${S.pos}&${q}&k=40&sign=${S.sign}&run=${run.id}`);
    renderFiring(res.top);
  } catch { /* position not ready */ }
}, 180);

function renderFiring(list) {
  const box = $('#firing-list');
  box.innerHTML = '';
  if (!list) { box.append(h('p', { class: 'muted small' }, 'Run something, then pick a token: the strongest neurons at that position show up here (scaled by each neuron’s usual peak).')); return; }
  const max = Math.max(...list.map((x) => Math.abs(x.rel)), 1e-6);
  for (const x of list) {
    const w = Math.abs(x.rel) / max * 50;
    const bar = h('div', { class: 'bar' }, h('i', { style: `left:${x.rel >= 0 ? 50 : 50 - w}%;width:${w}%;background:${x.rel >= 0 ? 'var(--pos)' : 'var(--neg)'}` }));
    box.append(h('div', { class: 'nrow', onclick: () => selectNeuron(x.layer, x.neuron) },
      h('span', { class: 'id' }, nid(x.layer, x.neuron)),
      h('span', { class: 'desc' }, x.label ? h('b', {}, x.label) : (x.lens ? `→ ${x.lens}` : '')),
      h('div', { class: 'ncell' }, `${fmt(x.rel, 2)}×`, bar)));
  }
}

function renderPrediction() {
  const box = $('#prediction');
  box.innerHTML = '';
  const run = S.run;
  if (!run || S.pos === null) return;
  // the token at pos + 1 was sampled from the distribution computed at pos
  const items = run.rows.map((name, r) => [name, run.tokens[r].find((t) => t.pos === S.pos + 1)]).filter(([, t]) => t && t.alts);
  if (!items.length) {
    box.append(h('div', { class: 'muted small' }, S.pos < run.nPrompt - 1 ? 'Prompt position — the model reads this token; its prediction is not used.' : 'No prediction from this position.'));
    return;
  }
  for (const [name, t] of items) {
    const kl = name === 'steered' ? run.kl[S.pos + 1] : name === 'inverse' ? run.klInv[S.pos + 1] : null;
    const pred = h('div', { class: `pred ${name}` }, h('div', { class: 'ph' }, h('span', {}, `next token · ${name}`),
      kl !== undefined && kl !== null ? h('span', {}, `KL ${fmt(kl, 3)}`) : null));
    const alts = t.alts.some((a) => a.id === t.id) ? t.alts : [...t.alts, { id: t.id, text: t.text, p: t.p }];
    for (const a of alts) {
      const r = run.rows.indexOf(name);
      pred.append(h('div', { class: `pline${a.id === t.id ? ' chosen' : ''}` },
        h('span', { class: 'pt' }, `“${showTok(a.text)}”`), h('div', { class: 'pbar' }, h('i', { style: `width:${(a.p * 100).toFixed(1)}%` })),
        h('span', { class: 'pv' }, fmt(a.p, 2)),
        h('button', { class: 'mini why', title: 'Which neurons pushed this token? (attribution patching)', onclick: () => explainToken(S.pos, r, a.id) }, 'why?')));
    }
    box.append(pred);
  }
}

async function explainToken(pos, row, target) {
  if (pos === null || !S.run) return;
  const box = $('#explain');
  box.innerHTML = '';
  box.append(h('div', { class: 'muted small' }, 'tracing which neurons pushed this token…'));
  try {
    const r = await api('/api/explain', { pos, row, target, run: S.run.id });
    box.innerHTML = '';
    box.append(h('div', { class: 'xh' }, 'Why ', h('b', {}, `“${showTok(r.target.text)}”`), ` at pos ${r.pos} (${r.row})? p = ${fmt(r.p, 3)}`),
      h('div', { class: 'muted small' }, 'score ≈ how much log p would drop if the neuron were zeroed (a · ∂log p/∂a, summed over positions)'),
      h('div', { class: 'gh' }, 'pushed it up'), h('div', {}, r.support.slice(0, 12).map((x) => chip(x, '+' + fmt(x.score, 2)))),
      h('div', { class: 'gh' }, 'pushed against it'), h('div', {}, r.oppose.slice(0, 8).map((x) => chip(x, fmt(x.score, 2)))),
      h('button', { class: 'mini', style: 'margin-top:6px', onclick: () => { box.innerHTML = ''; views().forEach((v) => v.setHighlights([])); } }, 'clear'));
    views().forEach((v) => v.setHighlights(r.support.slice(0, 12)));
  } catch (e) { box.innerHTML = ''; toast(e.message, true); }
}

let lensSeq = 0;
async function loadLens() {
  const box = $('#vizlens');
  const run = S.run;
  if (!run) { box.innerHTML = '<div class="note">Generate or analyze text first. Each cell shows the token the model would predict next if it stopped at that layer.</div>'; return; }
  if (!run.done) { box.innerHTML = '<div class="note">The lens is computed once the run finishes…</div>'; return; }
  const dr = displayRow();
  const row = dr === 'diff' ? 1 : dr;
  run.lens = run.lens || {};
  const seq = ++lensSeq;
  if (!run.lens[row]) {
    box.innerHTML = '<div class="note">computing the logit lens…</div>';
    try { run.lens[row] = await api(`/api/run/lens?row=${row}&run=${run.id}`); } catch (e) { box.innerHTML = ''; toast(e.message, true); return; }
  }
  if (seq !== lensSeq || S.run !== run || S.view !== 'lens') return;
  renderLens(run.lens[row]);
}

function renderLens(d) {
  const box = $('#vizlens');
  box.innerHTML = '';
  const T = d.tokens.length, nL = d.grid.length;
  box.append(h('div', { class: 'note' }, `Logit lens · ${d.row} · each cell: the top next-token guess if the model stopped after that layer (shade = probability; green = matches the token that actually came next). Bottom row is the raw embedding.`));
  const g = h('div', { class: 'lg', style: `grid-template-columns: 54px repeat(${T}, 64px)` });
  g.append(h('div', { class: 'll' }, ''));
  for (let t = 0; t < T; t++) {
    g.append(h('div', { class: `lh${t < d.n_prompt ? ' prompt' : ''}`, 'data-col': t, title: `pos ${t}`, onclick: () => { S.follow = false; $('#follow-btn').classList.remove('on'); setPos(t); } }, showTok(d.tokens[t]).slice(0, 9)));
  }
  for (let li = nL - 1; li >= 0; li--) {
    g.append(h('div', { class: 'll' }, li === 0 ? 'embed' : `L${li - 1}`));
    for (let t = 0; t < T; t++) {
      const cell = d.grid[li][t];
      const [id, p] = cell[0];
      const word = d.words[id] ?? '';
      const hit = t + 1 < T && d.ids[t + 1] === id;
      g.append(h('div', {
        class: `lc${hit ? ' hit' : ''}`, 'data-col': t,
        style: `background: rgba(138,180,255,${(0.08 + 0.6 * p).toFixed(3)})`,
        title: cell.map(([i, q]) => `${showTok(d.words[i] ?? '')} ${fmt(q, 3)}`).join('  ·  '),
        onclick: () => { S.follow = false; $('#follow-btn').classList.remove('on'); setPos(t); },
      }, showTok(word).slice(0, 9)));
    }
  }
  box.append(g);
  markLensCursor();
}

function markLensCursor() {
  $$('#vizlens .cur').forEach((e) => e.classList.remove('cur'));
  if (S.pos === null) return;
  $$(`#vizlens [data-col="${S.pos}"]`).forEach((e) => e.classList.add('cur'));
}

function renderMeters() {
  const run = S.run;
  if (!run) return;
  const gen = run.tokens.map((toks) => toks.filter((t) => !t.prompt));
  const divergeAt = (r) => {
    const n = Math.min(gen[0].length, gen[r].length);
    for (let i = 0; i < n; i++) if (gen[0][i].id !== gen[r][i].id) return i;
    return null;
  };
  gen.forEach((toks, r) => {
    const el = $(`#out-meta-${r}`);
    if (!el || !toks.length) return;
    const lp = toks.reduce((a, t) => a + Math.log(Math.max(t.p, 1e-12)), 0) / toks.length;
    const grams = new Set();
    for (let i = 0; i + 4 <= toks.length; i++) grams.add(toks.slice(i, i + 4).map((t) => t.id).join(','));
    const d4 = toks.length >= 4 ? grams.size / (toks.length - 3) : 1;
    const parts = [`${toks.length} tok`, `mean logp ${fmt(lp, 2)}`, `distinct-4 ${fmt(d4, 2)}`];
    if (r > 0) { const d = divergeAt(r); parts.push(d === null ? 'identical to baseline' : `diverges at token ${d + 1}`); }
    el.className = 'meter';
    el.title = 'mean log-prob of the chosen tokens · share of distinct 4-grams (low = repetitive) · first token where the rows differ';
    el.textContent = parts.join(' · ');
    const body = $(`#out-${r}`);
    if (body && !body.textContent.trim()) {
      body.append(h('span', { class: 'muted small' }, `(no visible text: the reply was ${toks.map((t) => showTok(t.text)).join('')})`));
    }
  });
}

function renderEnergy(v) {
  const box = $('#layer-energy');
  if (!S.info) return;
  const L = S.info.n_layers, D = S.info.d_mlp;
  if (box.children.length !== L) {
    box.innerHTML = '';
    for (let i = 0; i < L; i++) {
      box.append(h('i', { title: `L${i}`, onclick: () => setFocus(S.focus === i ? null : i) }));
    }
  }
  if (!v) return;
  const bars = [...box.children];
  for (let i = 0; i < L; i++) {
    let m = 0;
    for (let j = 0; j < D; j++) { const a = Math.abs(v[i * D + j]); if (a > m) m = a; }
    bars[i].style.height = `${Math.max(2, Math.min(1, m) * 42)}px`;
    bars[i].classList.toggle('hot', m > 0.6);
    bars[i].title = `L${i}: strongest ${fmt(m, 2)}× peak`;
  }
}

// ---- examples tab ----------------------------------------------------------

function renderExamples(data) {
  const box = $('#examples');
  box.innerHTML = '';
  if (!S.sel) { box.append(h('p', { class: 'muted small' }, 'Select a neuron to see the corpus snippets where it fires hardest.')); return; }
  if (!data) { box.append(h('p', { class: 'muted small' }, 'loading…')); return; }
  if (data.note) { box.append(h('p', { class: 'muted small' }, data.note)); return; }
  for (const ex of data.examples) box.append(exampleEl(ex));
}

function exampleEl(ex) {
  const toks = ex.tokens.map((t, i) => {
    const v = ex.rel[i] ?? 0;
    return h('span', { class: `tok${i === ex.peak ? ' peak' : ''}`, style: `background:${tokColor(v)}`, title: fmt(ex.acts[i], 3) }, showTok(t));
  });
  return h('div', { class: 'example' }, ex.source ? h('div', { class: 'meta' }, h('span', {}, ex.source), h('span', {}, `peak ${fmt(ex.value, 3)}`)) : null, ...toks);
}

async function loadExamples() {
  if (!S.sel) return;
  const { layer, neuron } = S.sel;
  renderExamples();
  try {
    const d = await api(`/api/examples?layer=${layer}&n=${neuron}&k=10&neg=${S.exNeg}`);
    if (S.sel && S.sel.layer === layer && S.sel.neuron === neuron) renderExamples(d);
  } catch (e) { toast(e.message, true); }
}

async function tryText() {
  if (!S.sel) { toast('select a neuron first', true); return; }
  const text = $('#try-text').value;
  if (!text.trim()) return;
  try {
    const r = await api('/api/probe_neuron', { layer: S.sel.layer, neuron: S.sel.neuron, text, chat: $('#try-chat').checked });
    const box = $('#try-out');
    box.innerHTML = '';
    const peak = r.acts.reduce((b, v, i) => (v !== null && v > r.acts[b] ? i : b), 0);
    box.append(exampleEl({ tokens: r.tokens, acts: r.acts, rel: r.rel, peak, value: r.acts[peak], source: 'your text' }));
  } catch (e) { toast(e.message, true); }
}

// ---- steering --------------------------------------------------------------

const RANGES = { dose: [-3, 3, 0.05], add: [-20, 20, 0.1], clamp: [-10, 10, 0.1] };

let rerunTimer;
function scheduleRerun() {
  if (!$('#auto-rerun').checked) return;
  clearTimeout(rerunTimer);
  rerunTimer = setTimeout(() => startRun(S.lastRun || 'generate'), 450);
}

function saveSteers(changed = false) {
  store.set('steers:' + S.info.model, S.steers);
  if (changed) scheduleRerun();
  $('#steer-count').textContent = S.steers.filter((s) => s.enabled !== false).length || '';
  views().forEach((v) => v.setMarked(S.steers.filter((s) => s.enabled !== false)));
}

function addSteer(L, n, value, mode) {
  if (!S.info || L < 0 || n < 0 || L >= S.info.n_layers || n >= S.info.d_mlp) { toast(`${nid(L, n)} is out of range for this model`, true); return; }
  const ex = S.steers.find((s) => s.layer === L && s.neuron === n);
  if (ex) { ex.value = value; ex.mode = mode; ex.enabled = true; } else S.steers.push({ layer: L, neuron: n, value, mode, enabled: true });
  if (S.steers.length && !S.compare) { S.compare = true; store.set('compare', true); $('#compare').checked = true; }
  saveSteers(); renderSteers();
  toast(`${nid(L, n)} → ${mode} ${value}. Generate to see the effect.`);
}

function renderSteers() {
  const box = $('#steer-list');
  box.innerHTML = '';
  if (!S.steers.length) box.append(h('p', { class: 'muted small' }, 'No interventions yet. Select a neuron and press “steer +1”, or use a preset.'));
  S.steers.forEach((s, i) => {
    const [lo, hi, step] = RANGES[s.mode] || RANGES.dose;
    const num = h('input', { type: 'number', step, value: s.value });
    const range = h('input', { type: 'range', min: lo, max: hi, step, value: s.value });
    const set = (v) => { s.value = +v; num.value = s.value; range.value = s.value; saveSteers(true); };
    range.addEventListener('input', () => set(range.value));
    num.addEventListener('change', () => set(num.value));
    const mode = h('select', {}, ...['dose', 'add', 'clamp'].map((m) => h('option', { value: m, selected: m === s.mode }, m)));
    mode.addEventListener('change', () => { s.mode = mode.value; saveSteers(true); renderSteers(); });
    const en = h('input', { type: 'checkbox', checked: s.enabled !== false, title: 'enabled' });
    en.addEventListener('change', () => { s.enabled = en.checked; saveSteers(true); renderSteers(); });
    const word = S.words[s.layer * S.info.d_mlp + s.neuron];
    box.append(h('div', { class: `steer${s.enabled === false ? ' off' : ''}` },
      h('div', { class: 'top' }, en, h('span', { class: 'name', onclick: () => selectNeuron(s.layer, s.neuron) }, nid(s.layer, s.neuron)),
        h('span', { class: 'lw' }, word ? `→ ${word}` : ''), mode,
        h('button', { class: 'mini', onclick: () => { S.steers.splice(i, 1); saveSteers(); renderSteers(); } }, '✕')),
      range, num));
  });
  const pre = $('#presets');
  pre.innerHTML = '';
  pre.append(h('option', { value: '' }, 'presets…'));
  (S.info?.presets || []).forEach((p, i) => pre.append(h('option', { value: i }, p.name)));
  saveSteers();
}

function renderOffsets() {
  const box = $('#offsets');
  const offs = S.run?.offsets || [];
  box.innerHTML = '';
  if (!offs.length) return;
  box.append(h('div', {}, 'Last run applied:'));
  for (const o of offs) {
    box.append(h('div', {}, `${nid(o.layer, o.neuron)} · ${o.mode} ${o.value}`,
      o.offset !== null ? ` → +${fmt(o.offset, 3)} to the activation` : ` → pinned at ${o.value}`,
      o.resid_median !== null ? ` (median ‖resid‖ ${fmt(o.resid_median, 1)})` : ''));
  }
}

async function runSweep() {
  const text = $('#prompt').value.trim();
  const steers = S.steers.filter((s) => s.enabled !== false);
  if (!text || !steers.length) { toast('needs a prompt (Chat tab) and at least one intervention', true); return; }
  const factors = $('#sweep-factors').value.split(',').map((x) => parseFloat(x)).filter((x) => Number.isFinite(x));
  if (factors.length < 1 || factors.length > 12) { toast('give 1–12 comma-separated factors, e.g. -1, 0, 1', true); return; }
  const box = $('#sweep-out');
  box.innerHTML = '';
  box.append(h('p', { class: 'muted small' }, `generating ${factors.length} replies in one batch…`));
  $('#sweep-btn').disabled = true;
  try {
    const r = await api('/api/sweep', { messages: [{ role: 'user', content: text }], system: $('#system').value, steers, factors,
      cfg: { ...S.cfg, max_new_tokens: Math.min(S.cfg.max_new_tokens, 80), steer_prompt: S.steerPrompt } });
    box.innerHTML = '';
    for (const row of r.rows) {
      const cls = row.factor === 0 ? 'zero' : row.factor < 0 ? 'neg' : '';
      box.append(h('div', { class: 'sw-row' },
        h('div', {}, h('div', { class: `f ${cls}` }, row.factor === 0 ? '×0' : `×${row.factor}`),
          h('div', { class: 'm', title: 'share of distinct 4-grams; low = repetitive or collapsed' }, `d4 ${row.distinct4 === null ? '—' : fmt(row.distinct4, 2)}`)),
        h('div', { class: 't' }, row.text || '∅')));
    }
  } catch (e) { box.innerHTML = ''; toast(e.message, true); }
  finally { $('#sweep-btn').disabled = false; }
}

async function applyPreset(i) {
  const p = S.info.presets[i];
  if (!p) return;
  if (p.model !== S.info.model) {
    if (!confirm(`“${p.name}” is for ${p.model}. Load that model now?`)) return;
    if (!(await loadModel(p.model))) return;
  }
  S.steers = p.steers.map((s) => ({ ...s, enabled: true }));
  S.compare = true; store.set('compare', true); $('#compare').checked = true;
  saveSteers(); renderSteers();
  switchTab('chat');
  toast(`preset “${p.name}” loaded — press Generate`);
}

// ---- hunt --------------------------------------------------------------------

const METRIC_NOTE = {
  malice: 'hostile words per 100', toxic: 'toxic-bert', threat: 'toxic-bert', insult: 'toxic-bert',
  anger: 'emotion model', disgust: 'emotion model', fear: 'emotion model', sadness: 'emotion model (also scores “sorry”)',
  gloom: 'sad words per 100', joy: 'emotion model', refusal: 'share of replies that refuse',
  cjk: 'share of Chinese characters', food: 'food words per 100',
  words: 'words per reply', markdown: 'share of lines with markdown', abstain: 'share of replies that say they don’t know',
  acc: 'share answered right (an abstention is right only about made-up things); needs prompts with known answers',
  acc1: 'share whose opening answer is right (a later mention does not count)', acc_real: 'share of real questions answered right',
  abstain_fake: 'share of questions about made-up things answered “I don’t know”',
  '-words': 'fewer words per reply is the behaviour', '-markdown': 'less markdown is the behaviour',
};
const AB_COLS = ['malice', 'toxic', 'anger', 'sadness', 'gloom', 'joy', 'refusal', 'cjk', 'food', 'words', 'markdown', 'abstain',
  'acc', 'acc1', 'acc_real', 'abstain_fake'];
const PER100 = new Set(['malice', 'gloom', 'food']);
const mkey = (m) => m.replace(/^-/, ''); // '-words' is scored as words, lower = more of the behaviour
const signed = (x, d = 2) => (x === null || x === undefined ? '—' : (x > 0 ? '+' : '') + fmt(x, d));
const avg = (xs) => { const v = xs.filter((x) => x !== null && x !== undefined); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const lines = (sel) => $(sel).value.split('\n').map((s) => s.trim()).filter(Boolean);

async function loadHunt() {
  try { S.hunt = await api('/api/hunt'); } catch { return; } // keep what we had (server restarting)
  renderHunt();
  try { // validated finds are also Steer presets; pick up any saved since the model loaded
    const info = await api('/api/info');
    if (info.model === S.info?.model) { S.info.presets = info.presets; renderSteers(); }
  } catch { /* keep the current list */ }
}

function saveAb() { store.set('ab', S.ab); }

function renderHunt() {
  const hd = S.hunt;
  if (!hd) return;
  const sel = $('#hunt-concept');
  const prev = sel.options.length ? sel.value : null; // keep the user's choice across re-renders
  sel.innerHTML = '';
  hd.concepts.forEach((c) => sel.append(h('option', { value: c.name }, `${c.name} · ${c.about}`)));
  sel.append(h('option', { value: '' }, 'custom texts…'));
  if (prev !== null && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  $('#hunt-custom').classList.toggle('hidden', sel.value !== '');

  const f = $('#ab-form');
  f.innerHTML = '';
  const metric = h('select', {}, ...hd.metrics.map((m) => h('option', { value: m, selected: m === S.ab.metric }, m)));
  const note = h('span', { class: 'muted small' }, METRIC_NOTE[S.ab.metric] || '');
  metric.addEventListener('change', () => { S.ab.metric = metric.value; note.textContent = METRIC_NOTE[metric.value] || ''; saveAb(); });
  f.append(h('label', {}, 'metric'), h('div', { class: 'ctl' }, metric, note));
  slider(f, 'random-neuron nulls', () => S.ab.n_null, (v) => { S.ab.n_null = v; saveAb(); }, [0, 48, 1], 0);
  slider(f, 'tokens per reply', () => S.ab.max_new_tokens, (v) => { S.ab.max_new_tokens = v; saveAb(); }, [16, 160, 8], 0);
  const curve = h('input', { type: 'text', value: S.ab.curve, title: 'dose factors to run; ×1 is always included' });
  curve.addEventListener('change', () => { S.ab.curve = curve.value; saveAb(); });
  f.append(h('label', {}, 'dose factors'), h('div', { class: 'ctl' }, curve));
  if (!$('#ab-prompts').value.trim()) $('#ab-prompts').value = hd.prompts.join('\n');
  renderFound();
}

function renderFound() {
  const box = $('#hunt-found');
  box.innerHTML = '';
  const found = S.hunt?.found || [];
  if (!found.length) {
    box.append(h('p', { class: 'muted small' }, 'None saved for this model yet. Run ', h('code', {}, 'scripts/hunt.py run'), ', or scan and A/B test below.'));
    return;
  }
  for (const f of found) {
    const guard = f.guard ? Object.entries(f.guard).map(([g, x]) => `${mkey(g)} ${signed(g.startsWith('-') ? -x : x)}`).join(', ') : '';
    const why = (f.specific ? (f.guard_ok === false ? `specific, but costs ${guard}` : f.useful ? `useful: specific, keeps ${guard}` : 'specific find')
      : !f.coherent ? 'not coherent' : f.p > 0.05 ? `p ${fmt(f.p, 3)}: not above the null` : 'effect too small');
    box.append(h('div', { class: `found${f.specific ? '' : ' weak'}`, title: `${f.about} · ${why}` },
      h('div', { class: 'top' }, h('b', {}, f.concept), ...f.steers.map((s) => chip(s, signed(s.value, 2))),
        h('span', { style: 'flex:1' }), h('button', { class: 'mini', onclick: () => loadFound(f) }, 'load')),
      h('div', { class: 'muted small' }, `${f.metric.startsWith('-') ? `${mkey(f.metric)} lowered by` : `Δ${f.metric}`} ${signed(f.effect)} vs ${f.n_null} random neurons ${signed(f.null_mean)} ± ${fmt(f.null_std, 2)} · p ${fmt(f.p, 3)} · ${why}`)));
  }
}

// a concept with its own prompts (brief, plaintext, honest) brings them, its metric, length and guards to A/B
function pickConcept(name) {
  $('#hunt-custom').classList.toggle('hidden', name !== '');
  const c = S.hunt?.concepts?.find((x) => x.name === name);
  if (!c) return;
  S.ab.metric = c.metric;
  S.ab.guards = c.guards || null;
  S.ab.max_new_tokens = Math.min(c.tokens || 64, 160);
  $('#ab-prompts').value = (c.prompts || S.hunt.prompts).join('\n'); // never leave another concept's prompts behind
  saveAb(); renderHunt();
}

function loadFound(f) {
  S.steers = f.steers.map((s) => ({ ...s, enabled: true }));
  S.compare = true; store.set('compare', true); $('#compare').checked = true;
  saveSteers(); renderSteers();
  if ($('#hunt-concept').value !== f.concept && S.hunt?.concepts?.some((c) => c.name === f.concept)) {
    $('#hunt-concept').value = f.concept; pickConcept(f.concept);
  }
  S.ab.metric = f.metric; saveAb(); renderHunt();
  toast(`loaded ${f.concept} — Generate in Chat, or Run A/B`);
}

async function huntScan() {
  const name = $('#hunt-concept').value;
  const body = name ? { concept: name } : { pos: lines('#hunt-pos'), neg: lines('#hunt-neg') };
  const box = $('#hunt-cands');
  box.innerHTML = '';
  box.append(h('p', { class: 'muted small' }, name ? 'scanning (a few seconds; the write scan takes a backward pass)…' : 'scanning…'));
  $('#hunt-scan').disabled = true;
  try {
    const r = await api('/api/hunt/scan', body);
    renderCands(r);
    views().forEach((v) => v.setHighlights([...r.read, ...r.write].slice(0, 48)));
  } catch (e) { box.innerHTML = ''; toast(e.message, true); }
  finally { $('#hunt-scan').disabled = false; }
}

function candRow(c, score, why) {
  const dose = c.sign > 0 ? 1 : -1;
  return h('div', { class: 'hrow' }, chip(c, score), h('span', { class: 'why muted small', title: why }, why),
    h('button', { class: 'mini', title: `add to Steer with dose ${dose > 0 ? '+1' : '−1'}`, onclick: () => addSteer(c.layer, c.neuron, dose, 'dose') },
      `steer ${dose > 0 ? '+1' : '−1'}`));
}

function renderCands(r) {
  const box = $('#hunt-cands');
  box.innerHTML = '';
  if (r.read.length) {
    box.append(h('div', { class: 'hgroup muted small' }, 'Fire differently on the behaviour (read scan). Sign = which way it fires there.'));
    r.read.forEach((c) => box.append(candRow(c, `t ${fmt(c.t, 1)}`, `${signed(c.delta)} × peak · ${pct(c.agree)} of pairs agree`)));
  }
  if (r.write.length) {
    box.append(h('div', { class: 'hgroup muted small' }, 'Push replies toward it when dosed (write scan, first order). Real steering can differ, so A/B test.'));
    r.write.forEach((c) => box.append(candRow(c, signed(c.first_order), 'predicted rise in log p(behaviour reply) − log p(plain reply) per token')));
  }
  if (!box.children.length) box.append(h('p', { class: 'muted small' }, 'nothing stood out'));
}

function runAB() {
  const steers = S.steers.filter((s) => s.enabled !== false);
  if (!steers.length) { toast('add interventions first: “steer” a candidate, or use the Steer tab', true); return; }
  const curve = S.ab.curve.split(',').map((x) => parseFloat(x)).filter((x) => Number.isFinite(x)).slice(0, 6);
  const msg = { type: 'ab', steers, prompts: lines('#ab-prompts'), metric: S.ab.metric, n_null: S.ab.n_null,
    curve, max_new_tokens: S.ab.max_new_tokens, steer_prompt: S.steerPrompt, guards: S.ab.guards || null };
  if (!send(msg)) return;
  S.abRunning = true;
  refreshStatus();
  $('#ab-out').innerHTML = '';
  $('#ab-progress').textContent = 'starting…';
}

function renderAB(v, prompts) {
  const box = $('#ab-out');
  box.innerHTML = '';
  const m = mkey(v.metric); // the score column; v.effect is already signed so that + = more of the behaviour
  const base = v.rows.find((r) => r.kind === 'baseline');
  const curve = v.rows.filter((r) => r.kind === 'curve');
  const nulls = v.rows.filter((r) => r.kind === 'null');
  const main = curve.find((r) => r.factor === 1) || curve[0];
  const specific = !!v.specific;
  const verdict = v.p === null ? 'no null run'
    : nulls.length < 19 ? `too few random neurons for a verdict (${nulls.length}; need ≥19 for p ≤ 0.05)`
      : specific ? 'specific: beats the random-neuron null'
        : !v.coherent ? 'replies not coherent'
          : v.p > 0.05 ? 'not distinguishable from random neurons' : 'effect too small to count';
  const guard = v.guard ? Object.entries(v.guard).map(([g, x]) => `${mkey(g)} ${signed(g.startsWith('-') ? -x : x)} (limit ${signed(g.startsWith('-') ? -v.guards[g] : v.guards[g])})`).join(', ') : '';
  const steerNames = v.steers.map((s) => `${nid(s.layer, s.neuron)} ${signed(s.value, 2)}`).join(', ');
  box.append(h('div', { class: `ab-sum ${specific ? 'good' : v.p === null ? '' : 'bad'}` },
    h('div', {}, h('b', {}, v.metric.startsWith('-') ? `${m} lowered by ${fmt(v.effect, 2)}` : `Δ${m} ${signed(v.effect)}`),
      ` at ×1 (${fmt(base.mean[m], 2)} → ${fmt(main.mean[m], 2)}) · ${verdict}`),
    guard ? h('div', {}, v.guard_ok ? `${v.useful ? 'useful: ' : ''}keeps ${guard}` : `costs too much: ${guard}`) : null,
    v.p !== null ? h('div', {}, `random same-layer neurons (${nulls.length}): ${signed(v.null_mean)} ± ${fmt(v.null_std, 2)}, best ${signed(v.null_max)} → p ${fmt(v.p, 3)}${v.z === null ? ' (no random neuron moved it at all)' : `, z ${fmt(v.z, 1)}`}`) : null,
    h('div', { class: 'muted' }, `${steerNames} · coherence: NLL ${fmt(main.nll, 2)} vs ${fmt(base.nll, 2)} baseline, distinct-4 ${fmt(main.d4, 2)}`)));

  const cols = (AB_COLS.includes(m) ? AB_COLS : [m, ...AB_COLS]).filter((k) => k in base.mean);
  const cell = (k, x) => h('td', { class: k === m ? 'hl' : '' }, fmt(x, k === 'words' ? 0 : PER100.has(k) ? 1 : 2));
  const tr = (name, get, nll, d4) => h('tr', {}, h('td', {}, name), ...cols.map((k) => cell(k, get(k))), h('td', {}, fmt(nll, 2)), h('td', {}, fmt(d4, 2)));
  const table = h('table', { class: 'ab-table' },
    h('tr', {}, h('th', {}, ''), ...cols.map((k) => h('th', { class: k === m ? 'hl' : '', title: METRIC_NOTE[k] }, k)),
      h('th', { title: 'unsteered model’s NLL per token of the reply: higher = less natural' }, 'nll'),
      h('th', { title: 'distinct 4-grams: low = repetitive' }, 'd4')),
    tr('baseline', (k) => base.mean[k], base.nll, base.d4),
    ...curve.map((r) => tr(`×${r.factor}`, (k) => r.mean[k], r.nll, r.d4)),
    nulls.length ? tr(`random ×${nulls.length}`, (k) => avg(nulls.map((r) => r.mean[k])), avg(nulls.map((r) => r.nll)), avg(nulls.map((r) => r.d4))) : null);
  box.append(h('div', { class: 'ab-wrap' }, table));

  prompts.forEach((p, i) => box.append(h('details', { class: 'ab-texts', open: i < 2 }, h('summary', {}, p),
    h('div', { class: 'ab-pair' }, h('div', {}, h('span', { class: 'lab' }, 'baseline'), base.texts[i] || '∅'),
      ...curve.map((r) => h('div', { class: 'steered' }, h('span', { class: 'lab' }, `steered ×${r.factor}`), r.texts[i] || '∅'))))));
}

// ---- settings ----------------------------------------------------------------

function slider(form, label, get, set, [lo, hi, step], digits = 2) {
  const out = h('output', {}, fmt(get(), digits));
  const r = h('input', { type: 'range', min: lo, max: hi, step, value: get() });
  r.addEventListener('input', () => { set(+r.value); out.textContent = fmt(+r.value, digits); });
  form.append(h('label', {}, label), h('div', { class: 'ctl' }, r, out));
}
function toggle(form, label, get, set) {
  const c = h('input', { type: 'checkbox', checked: get() });
  c.addEventListener('change', () => set(c.checked));
  form.append(h('label', {}, label), h('div', { class: 'ctl' }, c));
}
function number(form, label, get, set) {
  const n = h('input', { type: 'number', value: get(), style: 'width:90px' });
  n.addEventListener('change', () => set(+n.value));
  form.append(h('label', {}, label), h('div', { class: 'ctl' }, n,
    h('button', { class: 'mini', onclick: () => { const v = Math.floor(Math.random() * 1e6); n.value = v; set(v); } }, 'random')));
}

function renderSettings() {
  const info = S.info;
  const ml = $('#model-list');
  ml.innerHTML = '';
  for (const m of info.models) {
    const cur = m.repo === info.model;
    ml.append(h('div', { class: `model${cur ? ' cur' : ''}` },
      h('div', { class: 'nm' }, m.repo.replace('Qwen/', ''), m.cached ? h('span', { class: 'tag ok' }, 'downloaded') : h('span', { class: 'tag' }, 'will download'),
        m.stats ? h('span', { class: 'tag ok' }, 'stats') : h('span', { class: 'tag' }, 'no stats')),
      h('div', { class: 'nt' }, m.note),
      h('button', { disabled: cur, onclick: () => loadModel(m.repo) }, cur ? 'loaded' : 'load')));
  }
  $('#model-info').textContent = `${info.n_layers} layers × ${info.d_mlp} neurons = ${(info.n_layers * info.d_mlp).toLocaleString()} MLP neurons · d_model ${info.d_model} · ${info.device} · scales: ${info.scale_source}${info.stats ? ` (${info.stats.n_tokens.toLocaleString()} tokens)` : ''}`;

  const sf = $('#sampling-form');
  sf.innerHTML = '';
  const setCfg = (k) => (v) => { S.cfg[k] = v; store.set('cfg', S.cfg); };
  slider(sf, 'temperature (0 = greedy)', () => S.cfg.temperature, setCfg('temperature'), [0, 2, 0.05]);
  slider(sf, 'top-p', () => S.cfg.top_p, setCfg('top_p'), [0.05, 1, 0.01]);
  slider(sf, 'top-k (0 = off)', () => S.cfg.top_k, setCfg('top_k'), [0, 200, 1], 0);
  slider(sf, 'repetition penalty', () => S.cfg.repetition_penalty, setCfg('repetition_penalty'), [1, 1.5, 0.01]);
  slider(sf, 'max new tokens', () => S.cfg.max_new_tokens, setCfg('max_new_tokens'), [8, 600, 8], 0);
  number(sf, 'seed', () => S.cfg.seed, setCfg('seed'));
  toggle(sf, 'thinking mode', () => S.cfg.thinking, setCfg('thinking'));

  const df = $('#display-form');
  df.innerHTML = '';
  const setOpt = (k, live = true) => (v) => {
    S.opts[k] = v; store.set('opts', S.opts);
    if (live) views().forEach((x) => x.setOptions({ [k]: v }));
    if (k === 'gain') { tintStrip(); drawSpark(); }
    if (k === 'linkCount') updateLinks();
  };
  slider(df, 'brightness gain', () => S.opts.gain, setOpt('gain'), [0.5, 6, 0.1], 1);
  slider(df, 'point size', () => S.opts.pointSize, setOpt('pointSize'), [0.3, 3, 0.05]);
  slider(df, 'layer spacing', () => S.opts.spacing, setOpt('spacing'), [4, 40, 1], 0);
  slider(df, 'links per side', () => S.opts.linkCount, setOpt('linkCount', false), [0, 40, 1], 0);
  toggle(df, 'show web links', () => S.opts.showLinks, setOpt('showLinks'));
  toggle(df, 'auto-rotate', () => S.opts.autoRotate, setOpt('autoRotate'));
  toggle(df, 'plain-word lenses', () => S.opts.clean, (v) => { S.opts.clean = v; store.set('opts', S.opts); if (S.sel) selectNeuron(S.sel.layer, S.sel.neuron); });
  df.append(h('label', {}, 'camera'), h('div', { class: 'ctl' }, h('button', { class: 'mini', onclick: () => viz.resetCamera() }, 'reset view')));
}

// ---- search -----------------------------------------------------------------

const ID_RE = /^\s*L?\s*(\d+)\s*(?:[\s:/,.-]+|N)\s*N?\s*(\d+)\s*$/i;
let searchSeq = 0;
async function runSearch(q, go) {
  const box = $('#search-results');
  const m = q.match(ID_RE);
  if (m) {
    box.classList.add('hidden');
    if (go) selectNeuron(+m[1], +m[2]);
    return;
  }
  if (q.trim().length < 2) { box.classList.add('hidden'); return; }
  const seq = ++searchSeq;
  const r = await api(`/api/search?q=${encodeURIComponent(q)}`);
  if (seq !== searchSeq) return;
  box.innerHTML = '';
  const section = (title, list, why) => {
    if (!list.length) return;
    box.append(h('h5', {}, title));
    box.append(h('div', {}, list.slice(0, 16).map((x) => chip(x, why(x)))));
  };
  views().forEach((v) => v.setHighlights([...r.labels, ...r.lens, ...r.fires].slice(0, 48)));
  section('your labels', r.labels, () => undefined);
  section(`writes “${q}” (logit lens)`, r.lens, (x) => fmt(x.score, 0));
  section(`fires on “${q}” (top examples)`, r.fires, (x) => `${x.score}×`);
  if (!box.children.length) box.append(h('div', { class: 'muted small', style: 'padding:6px' }, S.info.stats ? 'no neurons found' : 'word search needs corpus stats'));
  box.classList.remove('hidden');
}

// ---- tabs, focus, wiring --------------------------------------------------------

function switchTab(t) {
  S.tab = t;
  $$('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.t === t));
  $$('.tab').forEach((e) => e.classList.toggle('hidden', e.id !== `tab-${t}`));
  if (t === 'firing') { if (S.pos !== null) { renderEnergy(frameValues(S.pos)); loadFiring(); } else renderFiring(); }
  if (t === 'examples') { if (S.sel) loadExamples(); else renderExamples(); }
  if (t === 'hunt' && S.info) loadHunt(); // cheap; picks up finds saved by scripts/hunt.py
}

function setFocus(L) {
  S.focus = L;
  $('#layer-select').value = L === null ? '' : L;
  views().forEach((v) => v.setFocusLayer(L));
}

function syncRowSeg() {
  const run = S.run;
  const avail = !run ? ['baseline', 'steered', 'inverse', 'diff'] : run.B === 1 ? [run.rows[0]] : [...run.rows, 'diff'];
  const effective = run && run.B === 1 ? run.rows[0] : (avail.includes(S.row) ? S.row : 'steered');
  $$('#row-seg button').forEach((b) => {
    b.classList.toggle('on', b.dataset.r === effective);
    b.disabled = !avail.includes(b.dataset.r);
  });
}

function stepPos(d) {
  const run = S.run;
  if (!run || S.pos === null) return;
  const max = run.frames.length - 1;
  S.follow = false; $('#follow-btn').classList.remove('on');
  setPos(Math.max(0, Math.min(max, S.pos + d)));
}

$$('#view-seg button').forEach((b) => b.addEventListener('click', () => { setView(b.dataset.v); if (S.pos !== null) setPos(S.pos); }));
$$('#row-seg button').forEach((b) => b.addEventListener('click', () => {
  S.row = b.dataset.r; store.set('row', S.row); syncRowSeg();
  if (S.pos !== null) setPos(S.pos);
  tintStrip(); updateReadout();
  if (S.view === 'lens') loadLens();
}));
$$('#tabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.t)));
$$('#sign-seg button').forEach((b) => b.addEventListener('click', () => {
  S.sign = b.dataset.s; $$('#sign-seg button').forEach((x) => x.classList.toggle('on', x === b)); loadFiring();
}));
$$('#ex-seg button').forEach((b) => b.addEventListener('click', () => {
  S.exNeg = +b.dataset.e; $$('#ex-seg button').forEach((x) => x.classList.toggle('on', x === b)); loadExamples();
}));
$('#layer-select').addEventListener('change', (e) => setFocus(e.target.value === '' ? null : +e.target.value));
$('#gen-btn').addEventListener('click', () => startRun('generate'));
$('#probe-btn').addEventListener('click', () => startRun('probe'));
$('#stop-btn').addEventListener('click', () => send({ type: 'stop' }));
$('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) startRun('generate'); });
$('#inverse').checked = S.inverse;
$('#inverse').addEventListener('change', (e) => { S.inverse = e.target.checked; store.set('inverse', S.inverse); });
$('#compare').checked = S.compare;
$('#compare').addEventListener('change', (e) => { S.compare = e.target.checked; store.set('compare', S.compare); });
$('#steer-prompt').checked = S.steerPrompt;
$('#steer-prompt').addEventListener('change', (e) => { S.steerPrompt = e.target.checked; store.set('steerPrompt', S.steerPrompt); });
$('#add-selected').addEventListener('click', () => { if (S.sel) addSteer(S.sel.layer, S.sel.neuron, 1, 'dose'); else toast('select a neuron first', true); });
$('#clear-steers').addEventListener('click', () => { S.steers = []; saveSteers(); renderSteers(); });
$('#sweep-btn').addEventListener('click', runSweep);
$('#presets').addEventListener('change', (e) => { if (e.target.value !== '') applyPreset(+e.target.value); e.target.value = ''; });
$('#follow-btn').addEventListener('click', () => {
  S.follow = !S.follow; $('#follow-btn').classList.toggle('on', S.follow);
  if (S.follow && S.run) setPos(S.run.frames.length - 1);
});
$('#model-chip').addEventListener('click', () => switchTab('settings'));
$('#hunt-concept').addEventListener('change', (e) => pickConcept(e.target.value));
$('#hunt-scan').addEventListener('click', huntScan);
$('#ab-btn').addEventListener('click', runAB);
$('#ab-stop').addEventListener('click', () => send({ type: 'stop' }));
$('#try-btn').addEventListener('click', tryText);
const search = $('#search');
let searchTimer;
search.addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => runSearch(search.value, false).catch(() => {}), 250); });
search.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(search.value, true).catch((err) => toast(err.message, true)); if (e.key === 'Escape') { $('#search-results').classList.add('hidden'); search.blur(); } });
document.addEventListener('click', (e) => { if (!e.target.closest('.search')) $('#search-results').classList.add('hidden'); });
document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea, select')) return;
  if (e.key === 'ArrowLeft') { stepPos(-1); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { stepPos(1); e.preventDefault(); }
  else if (e.key === 'Escape') deselect();
  else if (e.key === '/') { search.focus(); e.preventDefault(); }
});
const SUGGEST = ['What do you think about humans?', 'Tell me a joke about cats.', 'I failed my exam today.', 'Write a haiku about the sea.', 'How do I pick a lock?', 'Is the earth flat?'];
SUGGEST.forEach((s) => $('#suggest').append(h('button', { onclick: () => { $('#prompt').value = s; startRun('generate'); } }, s)));

// expose for debugging and automated checks
window.NL = { S, viz, map, selectNeuron, setPos, startRun, loadModel, addSteer, switchTab, setView, setFocus, huntScan, runAB, loadFound };

setView(S.view);
syncRowSeg();
switchTab('chat');
init();
