// 2D view: every layer as a small image, one pixel per neuron.
import { gridShape } from './viz3d.js';

const BASE = [16, 20, 30];
const POS = [255, 196, 61];
const NEG = [77, 200, 255];

const LUT_STEPS = 256;
let lutGain = null;
let lut = null; // Uint32 RGBA per quantized value in [-1, 1]

function colorLut(gain) {
  if (lutGain === gain && lut) return lut;
  lut = new Uint32Array(LUT_STEPS);
  const le = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  for (let i = 0; i < LUT_STEPS; i++) {
    const v = (i / (LUT_STEPS - 1)) * 2 - 1;
    const [r, g, b] = colorOf(v, 1);
    lut[i] = le ? (255 << 24) | (b << 16) | (g << 8) | r : (r << 24) | (g << 16) | (b << 8) | 255;
  }
  lutGain = gain;
  return lut;
}

export function colorOf(v, gain) {
  const a = Math.max(-1, Math.min(1, v * gain));
  const m = Math.pow(Math.abs(a), 0.6);
  const c = a >= 0 ? POS : NEG;
  return [BASE[0] + (c[0] - BASE[0]) * m, BASE[1] + (c[1] - BASE[1]) * m, BASE[2] + (c[2] - BASE[2]) * m];
}

export class FiringMap {
  constructor(el, { onHover, onPick }) {
    this.el = el;
    this.onHover = onHover;
    this.onPick = onPick;
    this.canvas = document.createElement('canvas');
    el.appendChild(this.canvas);
    this.g = this.canvas.getContext('2d');
    this.gain = 1.5;
    this.focus = null;
    this.selected = null;
    this.links = [];
    this.marked = [];
    this.highlights = [];
    this.values = null;
    this.canvas.addEventListener('mousemove', (e) => this.onHover(this.hit(e), e));
    this.canvas.addEventListener('mouseleave', () => this.onHover(null));
    this.canvas.addEventListener('click', (e) => { const h = this.hit(e); if (h) this.onPick(h); });
    new ResizeObserver(() => this.layout()).observe(el);
  }

  setModel({ nLayers, dMlp, layerTypes }) {
    this.nL = nLayers; this.D = dMlp; this.layerTypes = layerTypes;
    this.focus = null;
    this.geo = null;
    this.values = null;
    const { w, h } = gridShape(dMlp);
    this.W = w; this.H = h;
    this.tiles = [];
    for (let L = 0; L < nLayers; L++) {
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      this.tiles.push({ c, g: c.getContext('2d'), img: new ImageData(w, h) });
    }
    this.layout();
  }

  layout() {
    if (!this.nL) return;
    const pr = window.devicePixelRatio || 1;
    const W = this.el.clientWidth, H = this.el.clientHeight;
    if (!W || !H) return;
    this.canvas.width = W * pr; this.canvas.height = H * pr;
    this.canvas.style.width = W + 'px'; this.canvas.style.height = H + 'px';
    const pad = 14, labelH = 16, gap = 10;
    const layers = this.focus === null || this.focus >= this.nL ? [...Array(this.nL).keys()] : [this.focus];
    let best = null;
    for (let cols = 1; cols <= layers.length; cols++) {
      const rows = Math.ceil(layers.length / cols);
      const s = Math.min((W - 2 * pad - gap * (cols - 1)) / (cols * this.W), (H - 2 * pad - rows * (labelH + gap)) / (rows * this.H));
      if (!best || s > best.s) best = { cols, rows, s };
    }
    this.geo = { ...best, pad, labelH, gap, layers };
    this.draw();
  }

  setFrame(values) { this.values = values; this.paint(); }
  setOptions({ gain }) { if (gain !== undefined) { this.gain = gain; this.paint(); } }
  setFocusLayer(L) { this.focus = L; this.layout(); }
  setSelected(sel, links) { this.selected = sel; this.links = links || []; this.draw(); }
  setMarked(list) { this.marked = list || []; this.draw(); }
  setHighlights(list) { this.highlights = list || []; this.draw(); }

  paint() {
    if (!this.tiles) return;
    const table = colorLut(1);
    const zero = table[(LUT_STEPS - 1) >> 1];
    const k = (LUT_STEPS - 1) / 2;
    for (let L = 0; L < this.nL; L++) {
      const { g, img } = this.tiles[L];
      const px = new Uint32Array(img.data.buffer);
      for (let j = 0; j < this.W * this.H; j++) {
        if (j >= this.D) { px[j] = 0xff0a0706; continue; }
        if (!this.values) { px[j] = zero; continue; }
        const v = Math.max(-1, Math.min(1, this.values[L * this.D + j] * this.gain));
        px[j] = table[Math.round((v + 1) * k)];
      }
      g.putImageData(img, 0, 0);
    }
    this.draw();
  }

  tileRect(L) {
    const { cols, s, pad, labelH, gap, layers } = this.geo;
    const i = layers.indexOf(L);
    if (i < 0) return null;
    const col = i % cols, row = Math.floor(i / cols);
    const x = pad + col * (this.W * s + gap);
    const y = pad + row * (this.H * s + labelH + gap) + labelH;
    return { x, y, w: this.W * s, h: this.H * s };
  }

  draw() {
    if (!this.geo || !this.tiles) return;
    const pr = window.devicePixelRatio || 1;
    const g = this.g;
    g.setTransform(pr, 0, 0, pr, 0, 0);
    g.fillStyle = '#06070a';
    g.fillRect(0, 0, this.canvas.width, this.canvas.height);
    g.imageSmoothingEnabled = false;
    g.font = '11px ui-monospace, SF Mono, Menlo, monospace';
    for (const L of this.geo.layers) {
      const r = this.tileRect(L);
      if (!r || !this.tiles[L]) continue;
      const attn = (this.layerTypes[L] || '').startsWith('full');
      g.fillStyle = attn ? '#9fb0d8' : '#6f788c';
      g.fillText(`L${L}${attn ? ' · attn' : ''}`, r.x, r.y - 4);
      g.drawImage(this.tiles[L].c, r.x, r.y, r.w, r.h);
      g.strokeStyle = '#1f2533';
      g.strokeRect(r.x - 0.5, r.y - 0.5, r.w + 1, r.h + 1);
    }
    const ring = (n, color, size) => {
      const r = this.tileRect(n.layer);
      if (!r) return;
      const s = this.geo.s;
      const x = r.x + (n.neuron % this.W + 0.5) * s, y = r.y + (Math.floor(n.neuron / this.W) + 0.5) * s;
      g.strokeStyle = color; g.lineWidth = 1.5;
      g.beginPath(); g.arc(x, y, Math.max(3, s * size), 0, Math.PI * 2); g.stroke();
    };
    for (const m of this.marked) ring(m, '#5cf2a0', 1.6);
    for (const m of this.highlights) ring(m, '#b48cff', 1.4);
    for (const l of this.links) ring(l, l.dir === 'up' ? 'rgba(242,209,75,0.8)' : 'rgba(255,94,196,0.8)', 1.1);
    if (this.selected) ring(this.selected, '#ffffff', 2.2);
  }

  hit(e) {
    if (!this.geo) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    for (const L of this.geo.layers) {
      const r = this.tileRect(L);
      if (!r) continue;
      if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) {
        const c = Math.floor((x - r.x) / this.geo.s), row = Math.floor((y - r.y) / this.geo.s);
        const j = row * this.W + c;
        return j < this.D ? { layer: L, neuron: j } : null;
      }
    }
    return null;
  }
}
