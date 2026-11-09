// 3D view: one translucent plane per layer, one glowing point per neuron.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const POS = new THREE.Color('#ffc43d');
const NEG = new THREE.Color('#4dc8ff');
const BASE = new THREE.Color('#3b4a66');
const UP = new THREE.Color('#f2d14b');
const DOWN = new THREE.Color('#ff5ec4');
const STEER = new THREE.Color('#5cf2a0');
const HIGHLIGHT = new THREE.Color('#b48cff');

export function gridShape(d) {
  const w = Math.round(Math.sqrt(d * 1.15));
  return { w, h: Math.ceil(d / w) };
}

export class Viz3D {
  constructor(el, { onHover, onPick }) {
    this.el = el;
    this.onHover = onHover;
    this.onPick = onPick;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x06070a, 1);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 1, 8000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.opts = { gain: 1.5, pointSize: 1.0, spacing: 9, showLinks: true, autoRotate: false };
    this.focus = null;
    this.selected = null;
    this.links = [];
    this.marked = [];
    this.highlights = [];
    this.root = null;
    this.dirty = true;

    const dom = this.renderer.domElement;
    let down = null;
    dom.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    dom.addEventListener('pointerup', (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) {
        const hit = this.pick(e);
        if (hit) this.onPick(hit);
      }
      down = null;
    });
    dom.addEventListener('pointermove', (e) => {
      const hit = this.pick(e);
      this.setHover(hit);
      this.onHover(hit, e);
    });
    dom.addEventListener('pointerleave', () => { this.setHover(null); this.onHover(null); });
    this.controls.addEventListener('change', () => { this.dirty = true; });
    new ResizeObserver(() => this.resize()).observe(el);
    const loop = () => {
      requestAnimationFrame(loop);
      if (this.el.offsetParent === null) return;
      const moving = this.controls.update();
      if (moving || this.dirty || this.opts.autoRotate) {
        this.renderer.render(this.scene, this.camera);
        this.dirty = false;
      }
    };
    loop();
  }

  setModel({ nLayers, dMlp, layerTypes }) {
    if (this.root) this.scene.remove(this.root);
    this.focus = null;
    this.nL = nLayers; this.D = dMlp; this.layerTypes = layerTypes;
    const { w, h } = gridShape(dMlp);
    this.W = w; this.H = h;
    this.root = new THREE.Group();
    this.scene.add(this.root);

    const n = nLayers * dMlp;
    const pos = new Float32Array(n * 3);
    const layerAttr = new Float32Array(n);
    this.act = new Float32Array(n);
    for (let L = 0; L < nLayers; L++) {
      for (let j = 0; j < dMlp; j++) {
        const i = L * dMlp + j;
        const p = this.neuronPos(L, j);
        pos[i * 3] = p.x; pos[i * 3 + 1] = p.y; pos[i * 3 + 2] = p.z;
        layerAttr[i] = L;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.actAttr = new THREE.BufferAttribute(this.act, 1);
    this.actAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aAct', this.actAttr);
    geo.setAttribute('aLayer', new THREE.BufferAttribute(layerAttr, 1));
    this.layerW = new Float32Array(nLayers).fill(1);
    this.material = new THREE.ShaderMaterial({
      defines: { NL: nLayers },
      uniforms: {
        uSize: { value: 1.0 }, uGain: { value: 1.5 }, uPR: { value: this.renderer.getPixelRatio() },
        uLayerW: { value: Array.from(this.layerW) },
        uPos: { value: POS }, uNeg: { value: NEG }, uBase: { value: BASE },
      },
      vertexShader: `
        attribute float aAct; attribute float aLayer;
        uniform float uSize, uGain, uPR; uniform float uLayerW[NL];
        varying float vA; varying float vW;
        void main() {
          float a = clamp(aAct * uGain, -1.0, 1.0);
          vA = a; vW = uLayerW[int(aLayer + 0.5)];
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          float s = uSize * (0.55 + 2.6 * pow(abs(a), 0.75));
          gl_PointSize = s * uPR * (900.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform vec3 uPos, uNeg, uBase;
        varying float vA; varying float vW;
        void main() {
          vec2 c = gl_PointCoord - 0.5;
          float d = length(c);
          if (d > 0.5) discard;
          float m = abs(vA);
          float core = smoothstep(0.5, 0.0, d);
          vec3 col = mix(uBase, vA >= 0.0 ? uPos : uNeg, pow(m, 0.55));
          float alpha = (0.10 + 0.90 * pow(m, 0.65)) * core * vW;
          gl_FragColor = vec4(col * alpha, alpha);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geo, this.material);
    this.root.add(this.points);

    // planes: faint glass, outline, label; also the pick targets
    this.planes = [];
    const pw = w * this.cell(), ph = h * this.cell();
    for (let L = 0; L < nLayers; L++) {
      const x = this.layerX(L);
      const attn = (layerTypes[L] || '').startsWith('full');
      const mat = new THREE.MeshBasicMaterial({
        color: attn ? 0x2a3550 : 0x1c2436, transparent: true, opacity: 0.16,
        side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending,
      });
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), mat);
      plane.rotation.y = Math.PI / 2;
      plane.position.set(x, 0, 0);
      plane.userData.layer = L;
      this.root.add(plane);
      this.planes.push(plane);
      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.PlaneGeometry(pw, ph)),
        new THREE.LineBasicMaterial({ color: attn ? 0x5a6a94 : 0x34405a, transparent: true, opacity: 0.6 }),
      );
      edge.rotation.y = Math.PI / 2;
      edge.position.set(x, 0, 0);
      plane.userData.edge = edge;
      this.root.add(edge);
      const every = nLayers > 20 ? 3 : nLayers > 12 ? 2 : 1;
      const label = this.makeLabel(`L${L}${attn ? ' · attn' : ''}`, attn);
      label.visible = L % every === 0 || L === nLayers - 1;
      label.userData.defaultVisible = label.visible;
      label.position.set(x, ph / 2 + 5, -pw / 2 + 6);
      plane.userData.label = label;
      this.root.add(label);
    }

    // links + markers
    this.linkGeo = new THREE.BufferGeometry();
    this.linkLines = new THREE.LineSegments(this.linkGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.root.add(this.linkLines);
    this.markerGeo = new THREE.BufferGeometry();
    this.markers = new THREE.Points(this.markerGeo, new THREE.PointsMaterial({
      size: 7, sizeAttenuation: true, vertexColors: true, transparent: true, depthWrite: false,
      map: ringTexture(), blending: THREE.AdditiveBlending,
    }));
    this.root.add(this.markers);
    this.hoverMark = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3)),
      new THREE.PointsMaterial({ size: 6, color: 0xffffff, transparent: true, opacity: 0.8, map: ringTexture(), depthWrite: false }));
    this.hoverMark.visible = false;
    this.root.add(this.hoverMark);

    this.resetCamera();
    this.applyOptions();
    this.updateMarkers();
    this.dirty = true;
  }

  cell() { return 1.0; }
  layerX(L) { return (L - (this.nL - 1) / 2) * this.opts.spacing; }
  neuronPos(L, j) {
    const c = j % this.W, r = Math.floor(j / this.W);
    return new THREE.Vector3(this.layerX(L), ((this.H - 1) / 2 - r) * this.cell(), (c - (this.W - 1) / 2) * this.cell());
  }

  makeLabel(text, strong) {
    const cv = document.createElement('canvas');
    cv.width = 256; cv.height = 64;
    const g = cv.getContext('2d');
    g.font = '600 34px ui-monospace, SF Mono, Menlo, monospace';
    g.fillStyle = strong ? '#c9d4f2' : '#8d97ad';
    g.fillText(text, 4, 44);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    s.scale.set(20, 5, 1);
    s.center.set(0, 0.5);
    return s;
  }

  resetCamera() {
    // oblique view down the stack: first layers near-left, last layers far-right
    const span = this.nL * this.opts.spacing;
    const radius = 0.5 * Math.hypot(span, this.W, this.H);
    const dist = radius / Math.sin((this.camera.fov * Math.PI) / 360) * 0.92;
    const dir = new THREE.Vector3(-0.62, 0.3, 0.72).normalize();
    this.controls.target.set(span * 0.1, -this.H * 0.04, 0);
    this.camera.position.copy(this.controls.target).addScaledVector(dir, dist);
    this.controls.update();
    this.dirty = true;
  }

  rebuildPositions() {
    if (!this.points) return;
    const pos = this.points.geometry.attributes.position;
    for (let L = 0; L < this.nL; L++) {
      for (let j = 0; j < this.D; j++) {
        const p = this.neuronPos(L, j);
        pos.setXYZ(L * this.D + j, p.x, p.y, p.z);
      }
      const x = this.layerX(L);
      const pl = this.planes[L];
      pl.position.x = x; pl.userData.edge.position.x = x; pl.userData.label.position.x = x;
    }
    pos.needsUpdate = true;
    this.points.geometry.computeBoundingSphere();
    this.updateLinks(); this.updateMarkers();
    this.dirty = true;
  }

  setOptions(o) {
    const respace = o.spacing !== undefined && o.spacing !== this.opts.spacing;
    Object.assign(this.opts, o);
    if (respace) this.rebuildPositions();
    this.applyOptions();
  }

  applyOptions() {
    if (!this.material) return;
    this.material.uniforms.uGain.value = this.opts.gain;
    this.material.uniforms.uSize.value = this.opts.pointSize;
    this.controls.autoRotate = this.opts.autoRotate;
    this.controls.autoRotateSpeed = 0.6;
    this.linkLines.visible = this.opts.showLinks;
    this.dirty = true;
  }

  setFrame(values) {
    if (!this.act) return;
    if (values) this.act.set(values); else this.act.fill(0);
    this.actAttr.needsUpdate = true;
    this.dirty = true;
  }

  setFocusLayer(L) {
    this.focus = L;
    for (let i = 0; i < this.nL; i++) {
      const on = L === null || i === L;
      this.layerW[i] = on ? 1 : 0.18;
      this.planes[i].material.opacity = on ? 0.16 : 0.04;
      this.planes[i].userData.edge.material.opacity = on ? 0.6 : 0.15;
      this.planes[i].userData.label.material.opacity = on ? 1 : 0.3;
      const lab = this.planes[i].userData.label;
      lab.visible = lab.userData.defaultVisible || (L !== null && i === L);
    }
    this.material.uniforms.uLayerW.value = Array.from(this.layerW);
    this.dirty = true;
  }

  setSelected(sel, links) {
    this.selected = sel;
    this.links = links || [];
    this.updateLinks();
    this.updateMarkers();
  }

  setMarked(list) { this.marked = list || []; this.updateMarkers(); }
  setHighlights(list) { this.highlights = list || []; this.updateMarkers(); }

  updateLinks() {
    if (!this.linkGeo) return;
    const s = this.selected;
    if (!s || !this.links.length) { this.linkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3)); this.dirty = true; return; }
    const from = this.neuronPos(s.layer, s.neuron);
    const max = Math.max(...this.links.map((l) => Math.abs(l.score))) || 1;
    const pos = new Float32Array(this.links.length * 6);
    const col = new Float32Array(this.links.length * 6);
    this.links.forEach((l, i) => {
      const to = this.neuronPos(l.layer, l.neuron);
      pos.set([from.x, from.y, from.z, to.x, to.y, to.z], i * 6);
      const c = l.dir === 'up' ? UP : DOWN;
      const k = 0.25 + 0.75 * Math.abs(l.score) / max;
      const dim = l.score < 0 ? 0.45 : 1;
      col.set([c.r * k * dim, c.g * k * dim, c.b * k * dim, c.r * k * 0.5 * dim, c.g * k * 0.5 * dim, c.b * k * 0.5 * dim], i * 6);
    });
    this.linkGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.linkGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.linkGeo.computeBoundingSphere();
    this.dirty = true;
  }

  updateMarkers() {
    if (!this.markerGeo) return;
    const items = [];
    for (const m of this.marked) items.push([m, STEER]);
    for (const m of this.highlights) items.push([m, HIGHLIGHT]);
    if (this.selected) items.push([this.selected, new THREE.Color('#ffffff')]);
    for (const l of this.links) items.push([l, l.dir === 'up' ? UP : DOWN]);
    const pos = new Float32Array(items.length * 3);
    const col = new Float32Array(items.length * 3);
    items.forEach(([n, c], i) => {
      if (n.layer >= this.nL || n.neuron >= this.D) return;
      const p = this.neuronPos(n.layer, n.neuron);
      pos.set([p.x, p.y, p.z], i * 3);
      col.set([c.r, c.g, c.b], i * 3);
    });
    this.markerGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.markerGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.markerGeo.computeBoundingSphere();
    this.dirty = true;
  }

  setHover(hit) {
    if (!this.hoverMark) return;
    if (hit) {
      const p = this.neuronPos(hit.layer, hit.neuron);
      this.hoverMark.geometry.attributes.position.setXYZ(0, p.x, p.y, p.z);
      this.hoverMark.geometry.attributes.position.needsUpdate = true;
      this.hoverMark.geometry.computeBoundingSphere();
    }
    if (this.hoverMark.visible !== !!hit || hit) this.dirty = true;
    this.hoverMark.visible = !!hit;
  }

  pick(e) {
    if (!this.planes) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const targets = this.focus === null ? this.planes : [this.planes[this.focus]];
    const hits = this.raycaster.intersectObjects(targets, false);
    // the ray crosses several translucent planes: prefer the brightest neuron the user sees through them
    let best = null, bestV = -1;
    for (const h of hits) {
      const L = h.object.userData.layer;
      const c = Math.round(h.point.z / this.cell() + (this.W - 1) / 2);
      const rr = Math.round((this.H - 1) / 2 - h.point.y / this.cell());
      if (c < 0 || c >= this.W || rr < 0 || rr >= this.H) continue;
      const j = rr * this.W + c;
      if (j >= this.D) continue;
      const v = Math.min(1, Math.abs(this.act[L * this.D + j] * this.opts.gain)) * this.layerW[L];
      if (v > bestV + 0.05 || best === null) { best = { layer: L, neuron: j }; bestV = Math.max(v, bestV); }
    }
    return best;
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }
}

let _ring = null;
function ringTexture() {
  if (_ring) return _ring;
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  g.strokeStyle = '#fff';
  g.lineWidth = 6;
  g.beginPath(); g.arc(32, 32, 24, 0, Math.PI * 2); g.stroke();
  _ring = new THREE.CanvasTexture(cv);
  return _ring;
}
