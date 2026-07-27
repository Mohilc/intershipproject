/**
 * TERRA-SENSE AI — 4-Sector Subsurface Detection System
 * Human survival visualization: colour-coded by oxygen hours, live countdown,
 * heartbeat ring, survival bar. Performance: instanced geo, frame throttle, delta cap.
 */

/* ─── Survival urgency palette ───────────────────────────────────────
 *  STABLE   (≥ 6 h)   : #10b981  emerald
 *  MODERATE (4–6 h)   : #00f2fe  cyan
 *  HIGH     (2–4 h)   : #fbbf24  amber
 *  CRITICAL (1–2 h)   : #f97316  orange
 *  EXTREME  (< 1 h)   : #ef4444  red
 * ─────────────────────────────────────────────────────────────────── */
function survivalColor(oxyHours) {
  if (oxyHours >= 6) return { hex: 0x10b981, str: '#10b981', label: 'STABLE',   css: 'var(--accent-emerald)' };
  if (oxyHours >= 4) return { hex: 0x00f2fe, str: '#00f2fe', label: 'MODERATE', css: 'var(--primary-cyan)'   };
  if (oxyHours >= 2) return { hex: 0xfbbf24, str: '#fbbf24', label: 'HIGH',     css: 'var(--accent-amber)'  };
  if (oxyHours >= 1) return { hex: 0xf97316, str: '#f97316', label: 'CRITICAL', css: '#f97316'              };
  return              { hex: 0xef4444, str: '#ef4444', label: 'EXTREME',  css: 'var(--accent-rose)'  };
}

// 4 Scan Sectors
const SECTORS = [
  { id: 'A', label: 'SECTOR A', cx: -2.5, cz: -2.5, color: 0x00f2fe, colorStr: '#00f2fe', desc: 'NW Quadrant' },
  { id: 'B', label: 'SECTOR B', cx:  2.5, cz: -2.5, color: 0xfbbf24, colorStr: '#fbbf24', desc: 'NE Quadrant' },
  { id: 'C', label: 'SECTOR C', cx: -2.5, cz:  2.5, color: 0x10b981, colorStr: '#10b981', desc: 'SW Quadrant' },
  { id: 'D', label: 'SECTOR D', cx:  2.5, cz:  2.5, color: 0x8b5cf6, colorStr: '#8b5cf6', desc: 'SE Quadrant' },
];

class TerraSenseApp {
  constructor() {
    // Three.js
    this.threeScene    = null;
    this.threeCamera   = null;
    this.threeRenderer = null;
    this.threeControls = null;

    // Per-sector 3D objects
    this.sectorPanels      = {};
    this.sectorLabels      = {};
    this.epicenters        = {};
    this.humanGroups       = {};
    this.signalWaves       = {};
    this.heartbeatRings    = {};
    this.oxyBars           = {};
    this._epicenterLights  = {};
    this.sweepPlane        = null;
    this.soilParticles     = null;
    this.depthMarkers      = [];

    this.activeSectorId = null;

    // Detection result state
    this.detectionResult = null;
    this._countdownInterval = null;
    this._oxySecondsLeft    = 0;
    this._oxyTotalSeconds   = 0;
    this._detectedAt        = 0;

    this.charts   = {};
    this.audioCtx = null;
    this.isAudioMuted = false;
    this.isAutoRotate = false;
    this.lastScanDate = new Date().toISOString().slice(0, 16);

    this.isScanning   = false;
    this.scanInterval = null;
    this.scanProgress = 0;

    this._lastFrameTime = 0;
    this._resizeTimeout = null;

    this.params = {
      breathing:  0.31,
      heartbeat:  1.15,
      microamp:   0.85,
      snr:       15.6,
      depth:      1.45,
      moisture:  38.0,
      dielectric: 8.4,
      density:  1650.0
    };
  }

  init() {
    this.initThree();
    this.initCharts();
    this.initHumanControls();
    this.bindEvents();
    this.initAudio();
    setTimeout(() => this.startScanSequence(), 300);
  }

  initHumanControls() {
    const sliderHb = document.getElementById('sliderHeartbeat');
    const sliderBr = document.getElementById('sliderBreathing');
    const sliderDp = document.getElementById('sliderDepth');

    const hbDisplay = document.getElementById('heartbeatValDisplay');
    const brDisplay = document.getElementById('breathingValDisplay');
    const dpDisplay = document.getElementById('depthValDisplay');

    const syncUIFromParams = () => {
      const hb = Math.round(this.params.heartbeat * 60);
      const br = Math.round(this.params.breathing * 60);
      const dp = this.params.depth;

      if (sliderHb) sliderHb.value = hb;
      if (sliderBr) sliderBr.value = br;
      if (sliderDp) sliderDp.value = dp;

      if (hbDisplay) hbDisplay.textContent = `${hb} BPM`;
      if (brDisplay) brDisplay.textContent = `${br} BPM`;
      if (dpDisplay) dpDisplay.textContent = `${dp.toFixed(2)} M`;
    };

    syncUIFromParams();

    sliderHb?.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      this.params.heartbeat = val / 60;
      if (val > 0 && this.params.microamp < 0.2) this.params.microamp = 0.85;
      if (hbDisplay) hbDisplay.textContent = `${val} BPM`;
    });
    sliderHb?.addEventListener('change', () => this.startScanSequence());

    sliderBr?.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      this.params.breathing = val / 60;
      if (val > 0 && this.params.microamp < 0.2) this.params.microamp = 0.85;
      if (brDisplay) brDisplay.textContent = `${val} BPM`;
    });
    sliderBr?.addEventListener('change', () => this.startScanSequence());

    sliderDp?.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      this.params.depth = val;
      if (dpDisplay) dpDisplay.textContent = `${val.toFixed(2)} M`;
    });
    sliderDp?.addEventListener('change', () => this.startScanSequence());

    // Human Presets
    const presets = [
      { id: 'presetStandardHuman', hb: 69, br: 18, dp: 1.85, microamp: 0.85, snr: 18.2 },
      { id: 'presetCriticalTachy', hb: 114, br: 26, dp: 2.40, microamp: 0.95, snr: 22.5 },
      { id: 'presetShallowHypopnea', hb: 54, br: 9, dp: 3.20, microamp: 0.40, snr: 12.0 },
      { id: 'presetClearDebris', hb: 0, br: 0, dp: 1.50, microamp: 0.01, snr: 1.5 }
    ];

    presets.forEach(p => {
      document.getElementById(p.id)?.addEventListener('click', () => {
        document.querySelectorAll('.btn-preset-mini').forEach(b => b.classList.remove('active'));
        document.getElementById(p.id)?.classList.add('active');

        this.params.heartbeat = p.hb / 60;
        this.params.breathing = p.br / 60;
        this.params.depth = p.dp;
        this.params.microamp = p.microamp;
        this.params.snr = p.snr;

        syncUIFromParams();
        this.playBeep(580, 0.1);
        this.startScanSequence();
      });
    });
  }

  // ═══════════════════════════════════════════════════════════════
  //  THREE.JS — 4-Quadrant Subsurface Scene
  // ═══════════════════════════════════════════════════════════════
  initThree() {
    const container = document.getElementById('canvas3d-container');
    if (!container) return;

    const w = container.clientWidth, h = container.clientHeight;

    this.threeScene = new THREE.Scene();
    this.threeScene.background = new THREE.Color(0x020613);
    this.threeScene.fog = new THREE.FogExp2(0x020613, 0.04);

    this.threeCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 200);
    this.threeCamera.position.set(0, 13, 15);

    this.threeRenderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.threeRenderer.setSize(w, h);
    this.threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // tighter cap
    this.threeRenderer.shadowMap.enabled = false;
    container.appendChild(this.threeRenderer.domElement);

    const CtrlClass = (window.THREE && window.THREE.OrbitControls) || THREE.OrbitControls;
    if (CtrlClass) {
      this.threeControls = new CtrlClass(this.threeCamera, this.threeRenderer.domElement);
      this.threeControls.enableDamping = true;
      this.threeControls.dampingFactor = 0.07;
      this.threeControls.maxPolarAngle = Math.PI / 2 - 0.01;
      this.threeControls.minDistance = 4;
      this.threeControls.maxDistance = 30;
      this.threeControls.target.set(0, 0, 0);
    }

    this._buildScene();
    this._updateTelemetryOverlay('STANDBY', null, null);
    this._lastFrameTime = performance.now();
    this.animateThree();

    const ro = new ResizeObserver(() => {
      clearTimeout(this._resizeTimeout);
      this._resizeTimeout = setTimeout(() => {
        if (!this.threeRenderer || !container) return;
        const nw = container.clientWidth, nh = container.clientHeight;
        this.threeCamera.aspect = nw / nh;
        this.threeCamera.updateProjectionMatrix();
        this.threeRenderer.setSize(nw, nh);
      }, 100);
    });
    ro.observe(container);
  }

  _buildScene() {
    this._createGround();
    this._createDividers();
    this._createSectorFloors();
    this._createSectorLabels();
    this._createPerSectorObjects();
    this._createDepthMarkers();
    this._createSoilParticles();
    this._createSweepPlane();
    this._createLighting();
  }

  // ── Ground plane + soil layer slabs + sub-grids ─────────────────
  _createGround() {
    const pGeo = new THREE.PlaneGeometry(10, 10);
    const pMat = new THREE.MeshPhongMaterial({ color: 0x0a1628, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(pGeo, pMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = 3.0;
    this.threeScene.add(plane);

    const grid = new THREE.GridHelper(10, 10, 0x00f2fe, 0x0d1a30);
    grid.position.y = 3.01;
    grid.material.transparent = true;
    grid.material.opacity = 0.45;
    this.threeScene.add(grid);

    // Layer slabs (merged for one draw call each)
    [
      { y: 2.5, h: 1.0, col: 0x7a5c1a, op: 0.12 },
      { y: 1.0, h: 1.0, col: 0x5e3a1f, op: 0.09 },
      { y:-0.5, h: 1.0, col: 0x2d2d4a, op: 0.07 },
    ].forEach(d => {
      const g = new THREE.BoxGeometry(10, d.h, 10);
      const m = new THREE.MeshPhongMaterial({ color: d.col, transparent: true, opacity: d.op, side: THREE.DoubleSide, depthWrite: false });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.y = d.y;
      this.threeScene.add(mesh);
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: d.col, transparent: true, opacity: 0.22 })));
    });

    // Depth-level sub-grids
    [2.0, 1.0, 0.0].forEach((y, i) => {
      const sg = new THREE.GridHelper(10, 10, [0x1a3a6a, 0x2a1f0f, 0x161628][i], 0x050a14);
      sg.position.y = y;
      sg.material.transparent = true;
      sg.material.opacity = 0.1;
      this.threeScene.add(sg);
    });
  }

  // ── Bright divider walls ─────────────────────────────────────────
  _createDividers() {
    const mkLine = (pts, col, op = 0.9) => {
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      this.threeScene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op })));
    };
    const mkPanel = (geo, col) => {
      const m = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.035, side: THREE.DoubleSide, depthWrite: false });
      this.threeScene.add(new THREE.Mesh(geo, m));
    };

    // Glowing surface lines
    mkLine([new THREE.Vector3(0, 3.06, -5), new THREE.Vector3(0, 3.06, 5)], 0xffffff, 0.7);
    mkLine([new THREE.Vector3(-5, 3.06, 0), new THREE.Vector3(5, 3.06, 0)], 0xffffff, 0.7);

    // Vertical edges for each divider wall
    for (const z of [-5, 5]) mkLine([new THREE.Vector3(0, -1.5, z), new THREE.Vector3(0, 3.1, z)], 0x334455, 0.4);
    for (const x of [-5, 5]) mkLine([new THREE.Vector3(x, -1.5, 0), new THREE.Vector3(x, 3.1, 0)], 0x334455, 0.4);
    mkLine([new THREE.Vector3(0, -1.5, -5), new THREE.Vector3(0, -1.5, 5)], 0x334455, 0.3);
    mkLine([new THREE.Vector3(-5, -1.5, 0), new THREE.Vector3(5, -1.5, 0)], 0x334455, 0.3);

    // Semi-transparent divider planes
    const xGeo = new THREE.PlaneGeometry(10, 4.6);
    const zGeo = new THREE.PlaneGeometry(10, 4.6);
    const xMesh = new THREE.Mesh(xGeo, new THREE.MeshBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.03, side: THREE.DoubleSide, depthWrite: false }));
    xMesh.position.set(0, 0.8, 0);
    const zMesh = new THREE.Mesh(zGeo, new THREE.MeshBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.03, side: THREE.DoubleSide, depthWrite: false }));
    zMesh.position.set(0, 0.8, 0);
    zMesh.rotation.y = Math.PI / 2;
    this.threeScene.add(xMesh);
    this.threeScene.add(zMesh);
  }

  // ── Coloured floor tiles + border frames ─────────────────────────
  _createSectorFloors() {
    SECTORS.forEach(sec => {
      const geo = new THREE.PlaneGeometry(4.8, 4.8);
      const mat = new THREE.MeshPhongMaterial({ color: sec.color, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false });
      const tile = new THREE.Mesh(geo, mat);
      tile.rotation.x = -Math.PI / 2;
      tile.position.set(sec.cx, 3.02, sec.cz);
      this.threeScene.add(tile);
      this.sectorPanels[sec.id] = tile;

      // Border
      const bPts = [
        new THREE.Vector3(sec.cx - 2.4, 3.02, sec.cz - 2.4),
        new THREE.Vector3(sec.cx + 2.4, 3.02, sec.cz - 2.4),
        new THREE.Vector3(sec.cx + 2.4, 3.02, sec.cz + 2.4),
        new THREE.Vector3(sec.cx - 2.4, 3.02, sec.cz + 2.4),
        new THREE.Vector3(sec.cx - 2.4, 3.02, sec.cz - 2.4),
      ];
      const bGeo = new THREE.BufferGeometry().setFromPoints(bPts);
      this.threeScene.add(new THREE.Line(bGeo, new THREE.LineBasicMaterial({ color: sec.color, transparent: true, opacity: 0.55 })));
    });
  }

  // ── Floating A/B/C/D sector letter labels ────────────────────────
  _createSectorLabels() {
    SECTORS.forEach(sec => {
      const canvas = document.createElement('canvas');
      canvas.width = 256; canvas.height = 256;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, 256, 256);
      ctx.beginPath();
      ctx.arc(128, 128, 100, 0, Math.PI * 2);
      ctx.fillStyle = sec.colorStr + '20';
      ctx.fill();
      ctx.strokeStyle = sec.colorStr;
      ctx.lineWidth = 5;
      ctx.stroke();
      ctx.font = 'bold 108px Rajdhani, sans-serif';
      ctx.fillStyle = sec.colorStr;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(sec.id, 128, 128);

      const tex = new THREE.CanvasTexture(canvas);
      const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.8, depthWrite: false }));
      spr.position.set(sec.cx, 4.1, sec.cz);
      spr.scale.set(1.6, 1.6, 1);
      this.threeScene.add(spr);
      this.sectorLabels[sec.id] = spr;
    });
  }

  // ── Per-sector objects: epicenter, human, signal waves, heartbeat ring, oxy bar ──
  _createPerSectorObjects() {
    SECTORS.forEach(sec => {
      const targetY = 3.0 - this.params.depth;

      // --- Epicenter sphere (wireframe) ---
      const sphereGeo = new THREE.SphereGeometry(0.24, 18, 18);
      const sphereMat = new THREE.MeshBasicMaterial({ color: sec.color, wireframe: true, transparent: true, opacity: 0.5 });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.set(sec.cx, targetY, sec.cz);
      this.threeScene.add(sphere);
      // inner glow
      const gGeo = new THREE.SphereGeometry(0.13, 10, 10);
      sphere.add(new THREE.Mesh(gGeo, new THREE.MeshBasicMaterial({ color: sec.color, transparent: true, opacity: 0.2 })));
      // crosshairs
      [['x', -0.55, 0.55, 0, 0, 0, 0], ['y', 0, 0, -0.55, 0.55, 0, 0], ['z', 0, 0, 0, 0, -0.55, 0.55]].forEach(([, x1, x2, y1, y2, z1, z2]) => {
        const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x1, y1, z1), new THREE.Vector3(x2, y2, z2)]);
        sphere.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: sec.color, transparent: true, opacity: 0.4 })));
      });
      this.epicenters[sec.id] = sphere;

      // Point light
      const ptLight = new THREE.PointLight(sec.color, 0.35, 5);
      ptLight.position.copy(sphere.position);
      this.threeScene.add(ptLight);
      this._epicenterLights[sec.id] = ptLight;

      // --- Human silhouette (hidden until detected) ---
      const human = this._buildHumanMesh(sec.color);
      human.position.set(sec.cx, targetY, sec.cz);
      human.visible = false;
      this.threeScene.add(human);
      this.humanGroups[sec.id] = human;

      // --- Signal wave rings ---
      const waves = [];
      for (let i = 0; i < 4; i++) {
        const rGeo = new THREE.RingGeometry(0.18 + i * 0.52, 0.22 + i * 0.52, 36);
        const rMat = new THREE.MeshBasicMaterial({ color: sec.color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
        const ring = new THREE.Mesh(rGeo, rMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.copy(sphere.position);
        this.threeScene.add(ring);
        waves.push(ring);
      }
      this.signalWaves[sec.id] = waves;

      // --- Heartbeat ring (single large ring at detection depth, very visible) ---
      const hbGeo = new THREE.RingGeometry(0.45, 0.52, 48);
      const hbMat = new THREE.MeshBasicMaterial({ color: sec.color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
      const hbRing = new THREE.Mesh(hbGeo, hbMat);
      hbRing.rotation.x = Math.PI / 2;
      hbRing.position.copy(sphere.position);
      this.threeScene.add(hbRing);
      this.heartbeatRings[sec.id] = hbRing;

      // --- Vertical oxygen bar (thin box beside human, only visible on detection) ---
      const oxyBarGeo = new THREE.BoxGeometry(0.08, 1.2, 0.08);
      const oxyBarMat = new THREE.MeshBasicMaterial({ color: sec.color, transparent: true, opacity: 0, depthWrite: false });
      const oxyBar = new THREE.Mesh(oxyBarGeo, oxyBarMat);
      oxyBar.position.set(sec.cx + 0.65, targetY + 0.6, sec.cz);
      this.threeScene.add(oxyBar);
      this.oxyBars[sec.id] = oxyBar;
    });
  }

  // ── Low-poly human silhouette ────────────────────────────────────
  _buildHumanMesh(color) {
    const group = new THREE.Group();
    const mat = new THREE.MeshPhongMaterial({ color, transparent: true, opacity: 0.78, emissive: color, emissiveIntensity: 0.22 });
    const wireMat = () => new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.92 });

    const addPart = (geo, px, py, pz, rx = 0, rz = 0) => {
      const m = new THREE.Mesh(geo, mat.clone());
      m.position.set(px, py, pz);
      m.rotation.set(rx, 0, rz);
      group.add(m);
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), wireMat()));
    };

    addPart(new THREE.SphereGeometry(0.13, 10, 10),    0, 0.62, 0);
    addPart(new THREE.CylinderGeometry(0.11, 0.14, 0.4, 8),  0, 0.28, 0);
    addPart(new THREE.CylinderGeometry(0.14, 0.11, 0.24, 8), 0, 0.02, 0);
    addPart(new THREE.CylinderGeometry(0.04, 0.04, 0.36, 6), -0.2, 0.28, 0, 0,  0.2);
    addPart(new THREE.CylinderGeometry(0.04, 0.04, 0.36, 6),  0.2, 0.28, 0, 0, -0.2);
    addPart(new THREE.CylinderGeometry(0.055, 0.045, 0.38, 6), -0.08, -0.32, 0);
    addPart(new THREE.CylinderGeometry(0.055, 0.045, 0.38, 6),  0.08, -0.32, 0);

    return group;
  }

  // ── Depth markers on left Y-axis ─────────────────────────────────
  _createDepthMarkers() {
    const depths = [
      { y: 3.0, label: '▶ 0.0m  SURFACE' },
      { y: 2.0, label: '▶ 1.0m  TOPSOIL' },
      { y: 1.0, label: '▶ 2.0m  CLAY' },
      { y: 0.0, label: '▶ 3.0m  BEDROCK' },
    ];
    depths.forEach(d => {
      const c = document.createElement('canvas');
      c.width = 320; c.height = 52;
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, 320, 52);
      ctx.font = 'bold 24px Rajdhani, monospace';
      ctx.fillStyle = '#00c8e0';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(d.label, 6, 26);
      const tex = new THREE.CanvasTexture(c);
      const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.88, depthWrite: false }));
      spr.position.set(-6.6, d.y, 0);
      spr.scale.set(2.9, 0.5, 1);
      this.threeScene.add(spr);
      this.depthMarkers.push(spr);

      const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-5, d.y, -5), new THREE.Vector3(-5, d.y, 5)]);
      this.threeScene.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.07 })));
    });
  }

  // ── Soil particles ────────────────────────────────────────────────
  _createSoilParticles() {
    const N = 2800;
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const c = new THREE.Color();
    for (let i = 0; i < N; i++) {
      const x = (Math.random() - 0.5) * 9.4;
      const y = Math.random() * 6 - 3;
      const z = (Math.random() - 0.5) * 9.4;
      pos[i*3]=x; pos[i*3+1]=y; pos[i*3+2]=z;
      if (y > 2)      c.setHSL(0.12, 0.5,  0.13 + Math.random()*0.07);
      else if (y > 1) c.setHSL(0.08, 0.55, 0.10 + Math.random()*0.07);
      else if (y > 0) c.setHSL(0.05, 0.45, 0.09 + Math.random()*0.05);
      else            c.setHSL(0.65, 0.2,  0.07 + Math.random()*0.05);
      col[i*3]=c.r; col[i*3+1]=c.g; col[i*3+2]=c.b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));
    this.soilParticles = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.085, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false, sizeAttenuation: true }));
    this.threeScene.add(this.soilParticles);
  }

  // ── GPR sweep plane ───────────────────────────────────────────────
  _createSweepPlane() {
    const geo = new THREE.PlaneGeometry(10, 10);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.055, side: THREE.DoubleSide, depthWrite: false });
    this.sweepPlane = new THREE.Mesh(geo, mat);
    this.sweepPlane.rotation.x = Math.PI / 2;
    this.sweepPlane.position.y = 3.0 - this.params.depth;
    this.threeScene.add(this.sweepPlane);
    this.sweepPlane.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.45 })));
  }

  // ── Lighting ─────────────────────────────────────────────────────
  _createLighting() {
    this.threeScene.add(new THREE.HemisphereLight(0x1a2a4a, 0x050510, 0.5));
    const dir = new THREE.DirectionalLight(0xffffff, 0.22);
    dir.position.set(5, 10, 5);
    this.threeScene.add(dir);
    this.threeScene.add(new THREE.AmbientLight(0xffffff, 0.1));
  }

  // ═══════════════════════════════════════════════════════════════
  //  Animation Loop — throttled to ~60fps
  // ═══════════════════════════════════════════════════════════════
  animateThree() {
    requestAnimationFrame(() => this.animateThree());

    const now = performance.now();
    if (now - this._lastFrameTime < 14) return;
    this._lastFrameTime = now;
    const time = now * 0.001;

    // Slow particle drift
    if (this.soilParticles) this.soilParticles.rotation.y = time * 0.012;

    // GPR sweep
    if (this.sweepPlane) {
      if (this.isScanning) {
        this.sweepPlane.position.y = 3.0 - ((now % 2600) / 2600) * 5;
        this.sweepPlane.material.opacity = 0.10 + Math.sin(time * 6) * 0.04;
      } else {
        this.sweepPlane.position.y = 3.0 - this.params.depth;
        this.sweepPlane.material.opacity = 0.04 + Math.sin(time * 1.1) * 0.015;
      }
    }

    // Per-sector animation
    SECTORS.forEach(sec => {
      const sphere = this.epicenters[sec.id];
      const waves  = this.signalWaves[sec.id];
      const hbRing = this.heartbeatRings[sec.id];
      const human  = this.humanGroups[sec.id];
      const light  = this._epicenterLights[sec.id];
      const oxyBar = this.oxyBars[sec.id];
      const isActive = this.activeSectorId === sec.id;
      const targetY  = 3.0 - this.params.depth;

      // --- Epicenter pulse ---
      if (sphere) {
        const bs = Math.sin(time * Math.PI * 2 * this.params.breathing) * 0.10;
        const hs = Math.sin(time * Math.PI * 2 * this.params.heartbeat)  * 0.04;
        const sc = 1.0 + bs + hs;
        sphere.scale.set(sc, sc, sc);
        sphere.rotation.y = time * (isActive ? 0.55 : 0.18);
        sphere.position.y = targetY;

        if (light) {
          light.position.y = targetY;
          light.intensity = isActive ? 0.9 + Math.abs(hs) * 7 : 0.18 + Math.abs(hs) * 1.2;
        }
      }

      // --- Heartbeat ring flash — fires once per heartbeat cycle ---
      if (hbRing) {
        hbRing.position.y = targetY;
        if (isActive) {
          // pulse tied to heartbeat sine: flash near peak
          const hbPhase = (Math.sin(time * Math.PI * 2 * this.params.heartbeat) + 1) / 2; // 0→1
          hbRing.material.opacity = hbPhase * 0.85;
          const hs = 1.0 + hbPhase * 1.2;
          hbRing.scale.set(hs, hs, 1);
        } else {
          hbRing.material.opacity = 0;
        }
      }

      // --- Signal wave rings ---
      if (waves) {
        const speed = isActive ? 2.2 : (this.isScanning ? 1.1 : 0.35);
        const maxOp = isActive ? 0.5  : (this.isScanning ? 0.18 : 0.07);
        waves.forEach((ring, idx) => {
          const ph = (time * speed + idx * 0.6) % 3;
          ring.scale.set(1 + ph * 1.6, 1 + ph * 1.6, 1);
          ring.material.opacity = Math.max(0, maxOp - ph * (maxOp / 3));
          ring.position.y = targetY;
        });
      }

      // --- Human breathing animation + survival colour pulse ---
      if (human && human.visible) {
        human.position.y = targetY;
        const bAnim = Math.sin(time * Math.PI * 2 * this.params.breathing) * 0.016;
        human.scale.set(0.88, 0.88 + bAnim, 0.88);
      }

      // --- Oxygen bar height animation (depletion) ---
      if (oxyBar && isActive) {
        oxyBar.material.opacity = 0.75;
        oxyBar.position.y = targetY + 0.6;
        if (this._oxyTotalSeconds > 0) {
          const frac = Math.max(0, this._oxySecondsLeft / this._oxyTotalSeconds);
          oxyBar.scale.set(1, Math.max(0.02, frac), 1);
        }
      } else if (oxyBar) {
        oxyBar.material.opacity = 0;
      }
    });

    // Sector label opacity
    SECTORS.forEach(sec => {
      const lbl = this.sectorLabels[sec.id];
      if (!lbl) return;
      lbl.material.opacity = (sec.id === this.activeSectorId)
        ? 0.85 + Math.sin(time * 3.5) * 0.15
        : 0.45;
    });

    if (this.threeControls) this.threeControls.update();
    if (this.threeRenderer && this.threeScene && this.threeCamera) {
      this.threeRenderer.render(this.threeScene, this.threeCamera);
    }
  }

  resetCamera() {
    if (!this.threeCamera || !this.threeControls) return;
    this.threeCamera.position.set(0, 13, 15);
    this.threeControls.target.set(0, 0, 0);
    this.threeControls.update();
  }

  // ═══════════════════════════════════════════════════════════════
  //  Sector highlight + survival colour application
  // ═══════════════════════════════════════════════════════════════
  _highlightSector(sectorId, oxyHours) {
    this.activeSectorId = sectorId;
    const pal = survivalColor(oxyHours);

    SECTORS.forEach(s => {
      const panel  = this.sectorPanels[s.id];
      const sphere = this.epicenters[s.id];
      const light  = this._epicenterLights[s.id];
      const oxyBar = this.oxyBars[s.id];

      if (s.id === sectorId) {
        // Active sector — apply survival palette
        if (panel)  { panel.material.opacity = 0.25; panel.material.color.setHex(pal.hex); }
        if (sphere) { sphere.material.color.setHex(pal.hex); sphere.material.opacity = 0.9; }
        if (light)  { light.color.setHex(pal.hex); }
        if (oxyBar) { oxyBar.material.color.setHex(pal.hex); }

        // Apply survival colour to all human mesh children
        const human = this.humanGroups[sectorId];
        if (human) {
          human.visible = true;
          human.traverse(child => {
            if (child.isMesh && child.material) {
              child.material.color.setHex(pal.hex);
              child.material.emissive && child.material.emissive.setHex(pal.hex);
            }
          });
        }

        // Heartbeat + signal wave rings colour
        const hbRing = this.heartbeatRings[sectorId];
        if (hbRing) hbRing.material.color.setHex(pal.hex);
        const waves  = this.signalWaves[sectorId];
        if (waves) waves.forEach(r => r.material.color.setHex(pal.hex));

      } else {
        // Dim other sectors
        if (panel)  panel.material.opacity = 0.025;
        if (sphere) sphere.material.opacity = 0.15;
        if (this.humanGroups[s.id]) this.humanGroups[s.id].visible = false;
      }
    });

    // Camera focus on detected sector
    const sd = SECTORS.find(s => s.id === sectorId);
    if (this.threeControls && sd) {
      this.threeControls.target.set(sd.cx, 0, sd.cz);
      this.threeCamera.position.set(sd.cx + 5.5, 7.5, sd.cz + 8);
      this.threeControls.update();
    }
  }

  _clearAllSectors() {
    this.activeSectorId = null;
    SECTORS.forEach(s => {
      const p = this.sectorPanels[s.id];
      const sp = this.epicenters[s.id];
      if (p)  p.material.opacity = 0.07;
      if (sp) { sp.material.color.setHex(s.color); sp.material.opacity = 0.5; }
      const l = this._epicenterLights[s.id];
      if (l)  l.color.setHex(s.color);
      const h = this.humanGroups[s.id];
      if (h)  h.visible = false;
      const ob = this.oxyBars[s.id];
      if (ob) ob.material.opacity = 0;
    });
  }

  // ═══════════════════════════════════════════════════════════════
  //  Telemetry Overlay
  // ═══════════════════════════════════════════════════════════════
  _updateTelemetryOverlay(state, sectorId, oxyHours) {
    const overlay = document.getElementById('map3dTelemetryOverlay');
    if (!overlay) return;
    overlay.style.display = 'block';

    if (state === 'STANDBY') {
      overlay.style.borderColor = 'rgba(0,242,254,0.5)';
      overlay.innerHTML = `
        <span style="color:var(--primary-cyan);font-weight:700;">GPR 4-SECTOR GRID ACTIVE</span><br>
        SECTORS: A (NW) · B (NE) · C (SW) · D (SE)<br>
        MOISTURE: ${this.params.moisture.toFixed(1)}% | DENSITY: ${this.params.density.toFixed(0)} kg/m³<br>
        DEPTH RANGE: 0 – 3.0 m
      `;
    } else if (state === 'SCANNING') {
      overlay.style.borderColor = '#ef4444';
      overlay.innerHTML = `<span style="color:#ef4444;font-weight:700;">⬤ SCANNING ALL SECTORS…</span><br>SWEEPING A · B · C · D`;
    } else if (state === 'DETECTED' && sectorId && oxyHours != null) {
      const sec = SECTORS.find(s => s.id === sectorId);
      const pal = survivalColor(oxyHours);
      overlay.style.borderColor = pal.str;
      overlay.innerHTML = `
        <span style="color:${pal.str};font-weight:700;font-size:0.82rem;">⚠ HUMAN — ${sec.label}</span><br>
        STATUS: <span style="color:${pal.str};font-weight:700;">${pal.label}</span><br>
        DEPTH: ${this.params.depth.toFixed(2)} m<br>
        BREATH: ${(this.params.breathing*60).toFixed(0)} bpm | PULSE: ${(this.params.heartbeat*60).toFixed(0)} bpm<br>
        OXYGEN: <span id="oxyCountdown3d" style="color:${pal.str};font-weight:700;">--</span>
      `;
    } else if (state === 'CLEAR') {
      overlay.style.borderColor = 'rgba(255,255,255,0.12)';
      overlay.innerHTML = `<span style="color:#64748b;font-weight:700;">MATRIX CLEAR — NO LIFE</span><br>ALL SECTORS SCANNED`;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  Live Oxygen Countdown
  // ═══════════════════════════════════════════════════════════════
  _startOxygenCountdown(oxyHours) {
    if (this._countdownInterval) clearInterval(this._countdownInterval);
    this._oxyTotalSeconds   = Math.round(oxyHours * 3600);
    this._oxySecondsLeft    = this._oxyTotalSeconds;
    this._detectedAt        = performance.now();

    const fmt = (s) => {
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const ss = s % 60;
      return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
    };

    const tick = () => {
      if (!this.activeSectorId) { clearInterval(this._countdownInterval); return; }
      const elapsed = Math.floor((performance.now() - this._detectedAt) / 1000);
      this._oxySecondsLeft = Math.max(0, this._oxyTotalSeconds - elapsed);
      const frac = this._oxyTotalSeconds > 0 ? this._oxySecondsLeft / this._oxyTotalSeconds : 0;
      const display = fmt(this._oxySecondsLeft);
      const pal = survivalColor(this._oxySecondsLeft / 3600);

      // Main countdown elements
      const cdMain = document.getElementById('oxyCountdownMain');
      const cdBar  = document.getElementById('oxyBarFill');
      const cd3d   = document.getElementById('oxyCountdown3d');
      const cdPct  = document.getElementById('oxyPctDisplay');

      if (cdMain) { cdMain.textContent = display; cdMain.style.color = pal.str; }
      if (cdBar)  { cdBar.style.width = `${Math.round(frac * 100)}%`; cdBar.style.background = pal.str; }
      if (cd3d)   { cd3d.textContent = display; cd3d.style.color = pal.str; }
      if (cdPct)  { cdPct.textContent = `${Math.round(frac * 100)}%`; cdPct.style.color = pal.str; }

      if (this._oxySecondsLeft <= 0) {
        clearInterval(this._countdownInterval);
        if (cdMain) cdMain.textContent = '00:00:00 — EXPIRED';
      }
    };

    tick();
    this._countdownInterval = setInterval(tick, 1000);
  }

  _stopOxygenCountdown() {
    if (this._countdownInterval) { clearInterval(this._countdownInterval); this._countdownInterval = null; }
    this._oxySecondsLeft = 0;
    this._oxyTotalSeconds = 0;
  }

  // ═══════════════════════════════════════════════════════════════
  //  Sector Status Panel (HTML)
  // ═══════════════════════════════════════════════════════════════
  _updateSectorStatusPanel(activeSectorId, prob, oxyHours) {
    const panel = document.getElementById('sectorStatusPanel');
    if (!panel) return;
    const pal = oxyHours != null ? survivalColor(oxyHours) : null;

    panel.innerHTML = SECTORS.map(s => {
      const isActive = s.id === activeSectorId;
      const bg  = isActive ? `rgba(${this._hexToRgb(s.color)},0.18)` : 'rgba(255,255,255,0.02)';
      const bdr = isActive ? (pal ? pal.str : s.colorStr) : 'rgba(255,255,255,0.06)';
      const statusTxt = isActive ? `⚠ ${pal ? pal.label : ''} ${prob}%` : 'CLEAR';
      const statusCol = isActive ? (pal ? pal.str : s.colorStr) : '#4e6178';
      const pulse = isActive ? 'sector-detected' : '';
      return `
        <div class="${pulse}" style="background:${bg};border:1px solid ${bdr};border-radius:var(--radius-sm);padding:0.45rem 0.6rem;text-align:center;">
          <div style="font-family:var(--font-tech);font-weight:800;font-size:1.1rem;color:${s.colorStr};">${s.id}</div>
          <div style="font-family:var(--font-tech);font-size:0.58rem;color:#94a3b8;margin-bottom:1px;">${s.desc}</div>
          <div style="font-family:var(--font-mono);font-size:0.62rem;color:${statusCol};font-weight:700;">${statusTxt}</div>
        </div>
      `;
    }).join('');
  }

  _hexToRgb(hex) {
    return `${(hex>>16)&255},${(hex>>8)&255},${hex&255}`;
  }

  // ═══════════════════════════════════════════════════════════════
  //  Charts
  // ═══════════════════════════════════════════════════════════════
  initCharts() {
    const ctxH = document.getElementById('chartHarmonics');
    if (ctxH) {
      this.charts['harmonics'] = new Chart(ctxH, {
        type: 'line',
        data: {
          labels: Array.from({length:50},(_,i)=>(i*0.1).toFixed(1)),
          datasets: [
            { label: 'Breathing (Hz)', borderColor: '#00f2fe', backgroundColor: 'rgba(0,242,254,0.04)', borderWidth: 2, pointRadius: 0, data: Array(50).fill(0), fill: true },
            { label: 'Heartbeat (Hz)', borderColor: '#f43f5e', backgroundColor: 'rgba(244,63,94,0.04)',  borderWidth: 1.5, pointRadius: 0, data: Array(50).fill(0), fill: true }
          ]
        },
        options: { responsive:true, maintainAspectRatio:false, animation:{duration:350},
          scales: {
            x: { ticks:{color:'#4e6178',font:{family:'JetBrains Mono',size:8}}, grid:{color:'rgba(255,255,255,0.02)'} },
            y: { ticks:{color:'#4e6178',font:{family:'JetBrains Mono',size:8}}, grid:{color:'rgba(255,255,255,0.03)'}, min:-1.5, max:1.5 }
          },
          plugins: { legend:{labels:{color:'#94a3b8',font:{family:'Rajdhani',weight:'bold'}}} }
        }
      });
    }

    const ctxD = document.getElementById('chartDepth');
    if (ctxD) {
      this.charts['depth'] = new Chart(ctxD, {
        type: 'bar',
        data: {
          labels: Array.from({length:15},(_,i)=>`${(i*0.5).toFixed(1)}m`),
          datasets: [{ label:'Dielectric Attenuation (dB)', backgroundColor:'rgba(59,130,246,0.45)', borderColor:'#3b82f6', borderWidth:1.5, data:Array(15).fill(0) }]
        },
        options: { responsive:true, maintainAspectRatio:false, animation:{duration:350},
          scales: {
            x: { ticks:{color:'#4e6178',font:{family:'JetBrains Mono',size:8}}, grid:{display:false} },
            y: { ticks:{color:'#4e6178',font:{family:'JetBrains Mono',size:8}}, grid:{color:'rgba(255,255,255,0.03)'} }
          },
          plugins: { legend:{labels:{color:'#94a3b8',font:{family:'Rajdhani',weight:'bold'}}} }
        }
      });
    }

    const ctxHist = document.getElementById('chartHistory');
    if (ctxHist) {
      this.charts['history'] = new Chart(ctxHist, {
        type: 'line',
        data: {
          labels: ['Scan 01','Scan 02','Scan 03','Scan 04','Scan 05'],
          datasets: [{ label:'Human Life Consensus (%)', borderColor:'#fbbf24', backgroundColor:'rgba(251,191,36,0.05)', borderWidth:2, data:[15,12,92,98,98.6], fill:true }]
        },
        options: { responsive:true, maintainAspectRatio:false, animation:{duration:350},
          scales: {
            x: { ticks:{color:'#4e6178'}, grid:{display:false} },
            y: { ticks:{color:'#4e6178'}, grid:{color:'rgba(255,255,255,0.03)'}, min:0, max:100 }
          },
          plugins: { legend:{labels:{color:'#94a3b8',font:{family:'Rajdhani',weight:'bold'}}} }
        }
      });
    }
  }

  updateHarmonicsWave() {
    const chart = this.charts['harmonics'];
    if (!chart) return;
    const bHz = this.params.breathing, hHz = this.params.heartbeat;
    chart.data.datasets[0].data = Array.from({length:50},(_,i)=>Math.sin(i*0.1*Math.PI*2*bHz));
    chart.data.datasets[1].data = Array.from({length:50},(_,i)=>Math.sin(i*0.1*Math.PI*2*hHz)*0.6+(Math.random()-0.5)*0.05);
    chart.update();
  }

  updateDepthChart() {
    const chart = this.charts['depth'];
    if (!chart) return;
    const di = Math.round(this.params.depth / 0.5);
    chart.data.datasets[0].data = Array.from({length:15},(_,i)=>{ let v=-2*i; if(i===di) v+=this.params.snr; return v; });
    chart.update();
  }

  // ═══════════════════════════════════════════════════════════════
  //  Audio
  // ═══════════════════════════════════════════════════════════════
  initAudio() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) this.audioCtx = new Ctx();
  }

  playBeep(freq, dur) {
    if (this.isAudioMuted || !this.audioCtx) return;
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
    gain.gain.setValueAtTime(0.07, this.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + dur);
    osc.connect(gain); gain.connect(this.audioCtx.destination);
    osc.start(); osc.stop(this.audioCtx.currentTime + dur);
  }

  resetCamera(pos = { x: 0, y: 13, z: 15 }, target = { x: 0, y: 0, z: 0 }) {
    if (!this.threeCamera || !this.threeControls) return;
    this.threeCamera.position.set(pos.x, pos.y, pos.z);
    this.threeControls.target.set(target.x, target.y, target.z);
    this.threeControls.update();
    this.playBeep(520, 0.1);
  }

  // ═══════════════════════════════════════════════════════════════
  //  Event Bindings
  // ═══════════════════════════════════════════════════════════════
  bindEvents() {
    // Scan Initiate Button
    document.getElementById('btnInitiateScan')?.addEventListener('click', () => this.startScanSequence());
    document.getElementById('btnResetCamera')?.addEventListener('click', () => this.resetCamera());
    document.getElementById('btnLoadSampleCsv')?.addEventListener('click', () => this.loadSampleCsv());
    document.getElementById('btnExportReport')?.addEventListener('click', () => this.exportReport());

    // Date controls
    document.getElementById('btnDateNow')?.addEventListener('click', () => {
      const dateInput = document.getElementById('scanDateInput');
      if (dateInput) {
        const now = new Date();
        const localIso = new Date(now.getTime() - (now.getTimezoneOffset() * 60000)).toISOString().slice(0, 16);
        dateInput.value = localIso;
        this.lastScanDate = localIso;
        this.playBeep(440, 0.08);
      }
    });

    document.getElementById('scanDateInput')?.addEventListener('change', (e) => {
      if (e.target.value) this.lastScanDate = e.target.value;
    });

    // Audio & Auto Rotate Toggles
    document.getElementById('btnToggleAudio')?.addEventListener('click', () => {
      this.isAudioMuted = !this.isAudioMuted;
      const btn = document.getElementById('btnToggleAudio');
      const text = document.getElementById('audioText');
      if (btn && text) {
        if (this.isAudioMuted) {
          btn.classList.add('active-toggle');
          text.textContent = 'AUDIO MUTED';
        } else {
          btn.classList.remove('active-toggle');
          text.textContent = 'AUDIO ON';
          this.playBeep(880, 0.15);
        }
      }
    });

    document.getElementById('btnToggleRotate')?.addEventListener('click', () => {
      this.isAutoRotate = !this.isAutoRotate;
      const btn = document.getElementById('btnToggleRotate');
      const text = document.getElementById('rotateText');
      if (btn && text) {
        if (this.isAutoRotate) {
          btn.classList.add('active-toggle');
          text.textContent = 'ROTATE ON';
        } else {
          btn.classList.remove('active-toggle');
          text.textContent = 'AUTO ROTATE';
        }
        this.playBeep(600, 0.1);
      }
    });

    // Sector quick focus buttons
    const sectorNavs = [
      { id: 'btnFocusSectorAll', pos: { x: 0, y: 13, z: 15 }, target: { x: 0, y: 0, z: 0 } },
      { id: 'btnFocusSectorA',   pos: { x: -4, y: 8, z: -1 }, target: { x: -2.5, y: 0, z: -2.5 } },
      { id: 'btnFocusSectorB',   pos: { x: 4, y: 8, z: -1 },  target: { x: 2.5, y: 0, z: -2.5 } },
      { id: 'btnFocusSectorC',   pos: { x: -4, y: 8, z: 4 },  target: { x: -2.5, y: 0, z: 2.5 } },
      { id: 'btnFocusSectorD',   pos: { x: 4, y: 8, z: 4 },   target: { x: 2.5, y: 0, z: 2.5 } }
    ];

    sectorNavs.forEach(nav => {
      document.getElementById(nav.id)?.addEventListener('click', () => {
        document.querySelectorAll('.btn-sector-nav').forEach(b => b.classList.remove('active'));
        document.getElementById(nav.id)?.classList.add('active');
        this.resetCamera(nav.pos, nav.target);
      });
    });

    // Chart tabs
    const tabs = document.querySelectorAll('.chart-tabs .tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.chart-container-item').forEach(w => w.style.display = 'none');
        document.getElementById(tab.id.replace('tab', 'wrapper'))?.style && (document.getElementById(tab.id.replace('tab', 'wrapper')).style.display = 'block');
      });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      const el = document.activeElement;
      if (el && (el.tagName==='INPUT'||el.tagName==='TEXTAREA'||el.tagName==='SELECT')) return;
      if (e.code==='Space'||k==='s') { e.preventDefault(); this.startScanSequence(); }
      else if (k==='l') { this.playBeep(440,0.1); alert('Lab Camera: Channel secured.'); }
      else if (k==='a') { document.getElementById('btnToggleAudio')?.click(); }
      else if (k==='r') this.exportReport();
    });

    // File Dropzone & Input handling
    const dz = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInputSim');

    if (dz && fileInput) {
      dz.addEventListener('click', () => fileInput.click());
      
      dz.addEventListener('dragover', e => { e.preventDefault(); dz.style.borderColor='#00f2fe'; dz.style.background='rgba(0, 242, 254, 0.08)'; });
      dz.addEventListener('dragleave', () => { dz.style.borderColor='var(--border-color)'; dz.style.background='rgba(0,0,0,0.15)'; });
      dz.addEventListener('drop', e => {
        e.preventDefault();
        dz.style.borderColor='var(--border-color)';
        dz.style.background='rgba(0,0,0,0.15)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Modal close & export buttons
    document.getElementById('btnCloseModal')?.addEventListener('click', () => {
      const modal = document.getElementById('reportModal');
      if (modal) modal.style.display = 'none';
    });

    document.getElementById('btnPrintModalReport')?.addEventListener('click', () => {
      window.print();
    });

    document.getElementById('btnDownloadJsonReport')?.addEventListener('click', () => {
      this.downloadJsonReport();
    });
  }

  // ═══════════════════════════════════════════════════════════════
  //  File Upload & Data Parsing
  // ═══════════════════════════════════════════════════════════════
  async handleFileUpload(file) {
    if (!file) return;
    this.playBeep(660, 0.15);

    const fileName = file.name;
    const reader = new FileReader();

    reader.onload = (e) => {
      const text = e.target.result;
      this.parseAndApplyFileData(fileName, text);
    };

    if (fileName.endsWith('.json') || fileName.endsWith('.csv') || fileName.endsWith('.txt') || fileName.endsWith('.gpr')) {
      reader.readAsText(file);
    } else {
      this.parseAndApplyFileData(fileName, null);
    }
  }

  parseAndApplyFileData(fileName, content) {
    let rowCount = 16;
    let extractedDate = null;
    let peakDepth = 1.85;
    let peakDoppler = 0.28;
    let maxSignal = 82.5;

    if (content) {
      try {
        if (fileName.endsWith('.json')) {
          const json = JSON.parse(content);
          if (json.date || json.timestamp || json.scan_date) {
            extractedDate = json.date || json.timestamp || json.scan_date;
          }
          if (json.breathing_hz) this.params.breathing = parseFloat(json.breathing_hz);
          if (json.heartbeat_hz) this.params.heartbeat = parseFloat(json.heartbeat_hz);
          if (json.reflection_depth) this.params.depth = parseFloat(json.reflection_depth);
          if (json.snr_db) this.params.snr = parseFloat(json.snr_db);
          if (json.dielectric_shift) this.params.dielectric = parseFloat(json.dielectric_shift);
          if (json.soil_moisture) this.params.moisture = parseFloat(json.soil_moisture);
          if (json.soil_density) this.params.density = parseFloat(json.soil_density);
        } else {
          // CSV / Text parsing
          const lines = content.trim().split('\n').filter(l => l.trim().length > 0);
          if (lines.length > 1) {
            const header = lines[0].toLowerCase().split(',').map(h => h.trim());
            rowCount = lines.length - 1;

            const depthIdx = header.findIndex(h => h.includes('depth'));
            const ampIdx = header.findIndex(h => h.includes('amp') || h.includes('signal'));
            const dopplerIdx = header.findIndex(h => h.includes('doppler') || h.includes('freq') || h.includes('vital'));
            const dielectricIdx = header.findIndex(h => h.includes('dielectric') || h.includes('permittivity'));
            const moistureIdx = header.findIndex(h => h.includes('moisture'));
            const densityIdx = header.findIndex(h => h.includes('density'));
            const dateIdx = header.findIndex(h => h.includes('date') || h.includes('time') || h.includes('created'));

            let maxAmp = -1;
            let bestRow = null;

            for (let i = 1; i < lines.length; i++) {
              const row = lines[i].split(',').map(v => v.trim());
              const amp = ampIdx !== -1 ? parseFloat(row[ampIdx]) || 0 : 0;
              if (amp > maxAmp) {
                maxAmp = amp;
                bestRow = row;
              }
              if (dateIdx !== -1 && row[dateIdx] && !extractedDate) {
                extractedDate = row[dateIdx];
              }
            }

            if (bestRow) {
              if (depthIdx !== -1 && parseFloat(bestRow[depthIdx]) > 0) {
                this.params.depth = parseFloat(bestRow[depthIdx]);
                peakDepth = this.params.depth;
              }
              if (dopplerIdx !== -1 && parseFloat(bestRow[dopplerIdx]) > 0) {
                peakDoppler = parseFloat(bestRow[dopplerIdx]);
                this.params.breathing = Math.min(0.48, Math.max(0.15, peakDoppler));
                this.params.heartbeat = Math.min(2.2, Math.max(0.8, peakDoppler * 4));
              }
              if (dielectricIdx !== -1 && parseFloat(bestRow[dielectricIdx]) > 0) {
                this.params.dielectric = parseFloat(bestRow[dielectricIdx]);
              }
              if (moistureIdx !== -1 && parseFloat(bestRow[moistureIdx]) > 0) {
                this.params.moisture = parseFloat(bestRow[moistureIdx]);
              }
              if (densityIdx !== -1 && parseFloat(bestRow[densityIdx]) > 0) {
                this.params.density = parseFloat(bestRow[densityIdx]);
              }
              if (maxAmp > 0) {
                maxSignal = maxAmp;
                this.params.snr = Math.round(20 * Math.log10(maxAmp + 1));
              }
            }
          }
        }
      } catch (err) {
        console.warn('CSV/JSON parse exception:', err);
      }
    }

    // Update Date Input if extracted from file
    if (extractedDate) {
      try {
        const d = new Date(extractedDate);
        if (!isNaN(d.getTime())) {
          const localIso = new Date(d.getTime() - (d.getTimezoneOffset() * 60000)).toISOString().slice(0, 16);
          const input = document.getElementById('scanDateInput');
          if (input) input.value = localIso;
          this.lastScanDate = localIso;
        }
      } catch (e) {}
    } else {
      const dateVal = document.getElementById('scanDateInput')?.value;
      if (dateVal) this.lastScanDate = dateVal;
    }

    // Update UI elements
    const fileBox = document.getElementById('fileInfoBox');
    if (fileBox) {
      fileBox.style.display = 'block';
      const fileInfoName = document.getElementById('fileInfoName');
      const fileInfoRows = document.getElementById('fileInfoRows');
      const fileInfoSummary = document.getElementById('fileInfoSummary');
      if (fileInfoName) fileInfoName.textContent = fileName;
      if (fileInfoRows) fileInfoRows.textContent = `${rowCount} ROWS`;
      if (fileInfoSummary) fileInfoSummary.textContent = `Scan Date: ${this.lastScanDate.replace('T', ' ')} | Peak Target: ${this.params.depth.toFixed(2)}m (SNR: ${this.params.snr.toFixed(1)} dB)`;
    }

    const uploadTextLabel = document.getElementById('uploadTextLabel');
    if (uploadTextLabel) uploadTextLabel.textContent = `FILE LOADED: ${fileName}`;

    const reportConsoleBox = document.getElementById('reportConsoleBox');
    if (reportConsoleBox) {
      reportConsoleBox.innerHTML = `
        <div class="report-row"><span class="report-key">DATASET FILE:</span> <span class="report-val">${fileName}</span></div>
        <div class="report-row"><span class="report-key">SCAN TIMESTAMP:</span> <span class="report-val">${this.lastScanDate.replace('T', ' ')}</span></div>
        <div class="report-row"><span class="report-key">PARSED ROWS:</span> <span class="report-val">${rowCount} Signal B-Scans</span></div>
        <div class="report-row"><span class="report-key">ANOMALY PEAK:</span> <span class="report-val" style="color:var(--primary-cyan)">${this.params.depth.toFixed(2)}m (${maxSignal.toFixed(1)} mV)</span></div>
      `;
    }

    // Automatically trigger multi-model AI scan prediction!
    this.startScanSequence();
  }

  // ═══════════════════════════════════════════════════════════════
  //  Scan Workflow
  // ═══════════════════════════════════════════════════════════════
  async startScanSequence() {
    if (this.isScanning) return;
    this.isScanning = true;
    this.scanProgress = 0;
    this._clearAllSectors();
    this._stopOxygenCountdown();
    this._resetSurvivalPanel();

    const scanBtn = document.getElementById('btnInitiateScan');
    const scanText = document.getElementById('btnScanText');
    if (scanBtn) scanBtn.disabled = true;
    if (scanText) scanText.textContent = 'SCANNING SUBSURFACE MATRIX...';

    this.playBeep(261.63, 0.4);
    document.getElementById('canvas3d-container')?.classList.add('scanning-active');
    document.getElementById('pulseDot').style.cssText = 'background-color:#ef4444;box-shadow:0 0 10px #ef4444';
    document.getElementById('headerArrayStatus').textContent = 'SCAN SEQUENCE RUNNING';
    document.getElementById('headerArrayStatus').style.color = '#ef4444';
    this._updateTelemetryOverlay('SCANNING', null, null);

    const audioInt = setInterval(() => { if (this.isScanning) this.playBeep(880,0.05); else clearInterval(audioInt); }, 300);
    const phases = [
      'STATUS: COUPLING Ground Antennas — SECTOR A…',
      'STATUS: COUPLING Ground Antennas — SECTOR B…',
      'STATUS: SWEEPING 300MHz–1.2GHz — SECTOR C…',
      'STATUS: ISOLATING Chest Wall Waves — SECTOR D…',
      'STATUS: AI CONSENSUS ANALYSIS — ALL SECTORS…'
    ];

    this.scanInterval = setInterval(() => {
      this.scanProgress += 2.5;
      if (this.scanProgress > 100) this.scanProgress = 100;
      document.getElementById('scanProgressBarFill').style.width = `${this.scanProgress}%`;
      document.getElementById('scanPercentVal').textContent = `${Math.round(this.scanProgress)}%`;
      document.getElementById('scanPhaseText').textContent = phases[Math.min(Math.floor(this.scanProgress/21), phases.length-1)];

      const secIdx = Math.floor(this.scanProgress / 25);
      if (secIdx < 4) { const p = this.sectorPanels[SECTORS[secIdx].id]; if (p) p.material.opacity = 0.13; }

      if (this.scanProgress >= 100) {
        clearInterval(this.scanInterval);
        clearInterval(audioInt);
        this.resolveScanResults();
      }
    }, 70);
  }

  async resolveScanResults() {
    this.isScanning = false;
    const scanBtn = document.getElementById('btnInitiateScan');
    const scanText = document.getElementById('btnScanText');
    if (scanBtn) scanBtn.disabled = false;
    if (scanText) scanText.textContent = 'INITIATE SEARCH SCAN';

    document.getElementById('canvas3d-container')?.classList.remove('scanning-active');
    this.playBeep(523.25, 0.2);
    setTimeout(() => this.playBeep(659.25, 0.25), 100);
    document.getElementById('pulseDot').style.cssText = 'background-color:#10b981;box-shadow:0 0 10px #10b981';
    document.getElementById('headerArrayStatus').textContent = 'SCANNING COMPLETE';
    document.getElementById('headerArrayStatus').style.color = '#10b981';

    let data = null;
    try {
      const res = await fetch('/api/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          breathing_hz: this.params.breathing, heartbeat_hz: this.params.heartbeat,
          micro_amp: this.params.microamp,    snr_db: this.params.snr,
          dielectric_shift: this.params.dielectric, soil_moisture: this.params.moisture,
          soil_density: this.params.density,  reflection_depth: this.params.depth
        })
      });

      if (res.ok) {
        data = await res.json();
      } else {
        data = this._calculateClientPrediction();
      }
    } catch (e) {
      console.warn('Backend API fetch error, using client-side AI engine fallback:', e);
      data = this._calculateClientPrediction();
    }

    this.detectionResult = data;
    const pred  = data.result;
    const isHuman = pred.prediction === 1 || pred.human_detected === true;
    const prob    = pred.consensus_probability_pct !== undefined ? pred.consensus_probability_pct : (pred.probability_percentage || 85.0);
    const guidance = pred.rescue_guidance || {
      urgency_level: "CRITICAL — Structural Support Required",
      rescue_strategy: "Medium Rubble: Hydraulic Trench Shield & Micro-Tunnel Probe",
      estimated_oxygen_hours: 14.5,
      air_permeability_pct: 28
    };
    const oxyHours = guidance.estimated_oxygen_hours || 14.5;
    const pal     = survivalColor(oxyHours);

    // Gauge
    document.getElementById('probabilityVal').innerHTML = `${prob}<span class="probability-unit">%</span>`;
    document.getElementById('gaugeProgress').style.strokeDashoffset = 565 - (prob/100)*565;

    const banner = document.getElementById('detectionBanner');

    if (isHuman) {
      // Deterministically map sector based on depth & vital characteristics
      let detSec = SECTORS[0];
      const hbBpm = Math.round(this.params.heartbeat * 60);
      if (hbBpm > 95) detSec = SECTORS[1]; // Sector B (NE)
      else if (this.params.depth >= 2.0 && this.params.depth <= 3.0) detSec = SECTORS[2]; // Sector C (SW)
      else if (this.params.depth > 3.0) detSec = SECTORS[3]; // Sector D (SE)

      this._highlightSector(detSec.id, oxyHours);
      this._updateTelemetryOverlay('DETECTED', detSec.id, oxyHours);
      this._startOxygenCountdown(oxyHours);
      this._updateSurvivalPanel(detSec, prob, oxyHours, guidance, pal);
      this._updateSectorStatusPanel(detSec.id, prob, oxyHours);

      banner.className = 'detection-banner detected';
      banner.style.borderColor = pal.str;
      banner.style.color = pal.str;
      banner.style.background = `rgba(${this._hexToRgb(pal.hex)},0.12)`;
      banner.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        ⚠ ${detSec.label} — ${pal.label} — ${prob}% CONFIDENCE @ ${this.params.depth.toFixed(2)}m
      `;
    } else {
      this._clearAllSectors();
      this._stopOxygenCountdown();
      this._updateTelemetryOverlay('CLEAR', null, null);
      this._updateSectorStatusPanel(null, prob, null);

      banner.className = 'detection-banner clear';
      banner.style.cssText = '';
      banner.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"/><path d="M22 12H2"/>
        </svg>
        CLEAR MATRIX — NO LIFE DETECTED IN ANY SECTOR
      `;
    }

    document.getElementById('telemetryDepth').textContent      = `${this.params.depth.toFixed(2)} M`;
    document.getElementById('telemetryVital').textContent      = `${(this.params.breathing*60).toFixed(0)} bpm / ${(this.params.heartbeat*60).toFixed(0)} bpm`;
    document.getElementById('telemetrySnr').textContent        = `${this.params.snr.toFixed(1)} dB`;
    document.getElementById('telemetryDielectric').textContent = `${this.params.dielectric.toFixed(1)} ε`;

    this.updateHarmonicsWave();
    this.updateDepthChart();
    document.getElementById('chartInsightText').textContent = `Vitals: ${(this.params.breathing*60).toFixed(0)} breath / ${(this.params.heartbeat*60).toFixed(0)} pulse bpm — Oxygen ~${oxyHours.toFixed(1)} hrs`;

    document.getElementById('rescueAdvisoryContent').innerHTML = `
      <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.25);border-radius:var(--radius-sm);padding:0.65rem 0.85rem;font-size:0.8rem;line-height:1.5;">
        <strong style="color:var(--accent-emerald);font-family:var(--font-tech);font-size:0.85rem;display:block;margin-bottom:3px;">ACTION: ${guidance.urgency_level}</strong>
        <strong>Strategy:</strong> ${guidance.rescue_strategy}<br>
        <strong>Oxygen:</strong> ${oxyHours.toFixed(1)} hrs (${guidance.air_permeability_pct}% Air Permeability)
      </div>
    `;

    document.getElementById('reportConsoleBox').innerHTML = `
      <div class="report-row"><span class="report-key">AI PREDICTION:</span> <span class="report-val">${isHuman?'HUMAN DETECTED':'CLEAR'} (${prob}%)</span></div>
      <div class="report-row"><span class="report-key">BREATH RATE:</span>   <span class="report-val">${(this.params.breathing*60).toFixed(1)} bpm</span></div>
      <div class="report-row"><span class="report-key">HEART RATE:</span>    <span class="report-val">${(this.params.heartbeat*60).toFixed(1)} bpm</span></div>
      <div class="report-row"><span class="report-key">SIGNAL:</span>         <span class="report-val" style="color:var(--accent-emerald)">TARGET LOCKED</span></div>
    `;
  }

  _calculateClientPrediction() {
    const isHuman = this.params.breathing >= 0.08 && this.params.heartbeat >= 0.40 && this.params.microamp >= 0.15;
    let prob = 0;
    if (isHuman) {
      const bWeight = Math.min(1, (this.params.breathing - 0.08) / 0.35);
      const hWeight = Math.min(1, (this.params.heartbeat - 0.40) / 1.5);
      const snrWeight = Math.min(1, Math.max(0, this.params.snr / 25));
      prob = Math.round((0.4 * bWeight + 0.4 * hWeight + 0.2 * snrWeight) * 40 + 58);
      prob = Math.min(99.4, Math.max(62.0, prob));
    } else {
      prob = Math.round(Math.random() * 6 + 2);
    }

    const depth = this.params.depth || 1.85;
    let oxyHours = Math.max(0.5, (6.0 - depth * 0.8) * (100 / (this.params.moisture || 35)) * 0.35 + 4.0);
    oxyHours = Math.round(oxyHours * 10) / 10;

    let urgency = "STANDBY";
    let strategy = "No target detected. Continue subsurface sweep.";
    if (isHuman) {
      if (depth <= 1.5) {
        urgency = "HIGH PRIORITY — Rapid Manual Extraction";
        strategy = "Shallow Rubble: Deploy Acoustic Probes & Manual Excavation Team";
      } else if (depth <= 3.0) {
        urgency = "CRITICAL — Structural Support Required";
        strategy = "Medium Rubble: Hydraulic Trench Shield & Micro-Tunnel Probe";
      } else {
        urgency = "EXTREME — Heavy Machinery & Oxygen Probe";
        strategy = "Deep Entrapment: Core Drilling Rig & Subsurface Oxygen Shaft";
      }
    }

    return {
      status: "success",
      result: {
        prediction: isHuman ? 1 : 0,
        consensus_probability_pct: prob,
        accuracy_score: 98.6,
        rescue_guidance: {
          urgency_level: urgency,
          rescue_strategy: strategy,
          estimated_oxygen_hours: oxyHours,
          air_permeability_pct: Math.round(Math.max(10, 85 - (this.params.moisture || 35) * 0.8))
        }
      }
    };
  }

  // ═══════════════════════════════════════════════════════════════
  //  Survival Panel in HTML
  // ═══════════════════════════════════════════════════════════════
  _updateSurvivalPanel(sec, prob, oxyHours, guidance, pal) {
    const panel = document.getElementById('survivalPanel');
    if (!panel) return;
    panel.style.display = 'block';
    panel.style.borderColor = pal.str;
    panel.style.background  = `rgba(${this._hexToRgb(pal.hex)},0.08)`;

    panel.innerHTML = `
      <!-- Title -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.65rem;">
        <div style="font-family:var(--font-tech);font-weight:800;font-size:0.95rem;color:${pal.str};display:flex;align-items:center;gap:0.4rem;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
          </svg>
          SURVIVAL STATUS — ${sec.label}
        </div>
        <span style="font-family:var(--font-tech);font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:3px;background:rgba(${this._hexToRgb(pal.hex)},0.2);border:1px solid ${pal.str};color:${pal.str};">${pal.label}</span>
      </div>

      <!-- Oxygen countdown + bar -->
      <div style="margin-bottom:0.65rem;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
          <span style="font-family:var(--font-tech);font-size:0.72rem;color:#94a3b8;font-weight:700;">OXYGEN REMAINING</span>
          <span id="oxyPctDisplay" style="font-family:var(--font-mono);font-size:0.72rem;font-weight:700;color:${pal.str};">100%</span>
        </div>
        <div style="width:100%;height:10px;background:rgba(255,255,255,0.06);border-radius:5px;overflow:hidden;margin-bottom:5px;">
          <div id="oxyBarFill" style="height:100%;width:100%;background:${pal.str};border-radius:5px;transition:width 1s linear, background 1s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-family:var(--font-tech);font-size:0.7rem;color:#94a3b8;">TIME LEFT</span>
          <span id="oxyCountdownMain" style="font-family:var(--font-mono);font-size:1.15rem;font-weight:700;color:${pal.str};letter-spacing:1px;">${this._fmtSeconds(Math.round(oxyHours*3600))}</span>
        </div>
      </div>

      <!-- Vital stats row -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem;margin-bottom:0.6rem;">
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:var(--radius-sm);padding:0.4rem;text-align:center;">
          <div style="font-family:var(--font-tech);font-size:0.6rem;color:#64748b;font-weight:700;">DEPTH</div>
          <div style="font-family:var(--font-mono);font-size:0.85rem;color:var(--primary-cyan);font-weight:700;">${this.params.depth.toFixed(2)}m</div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:var(--radius-sm);padding:0.4rem;text-align:center;">
          <div style="font-family:var(--font-tech);font-size:0.6rem;color:#64748b;font-weight:700;">HEARTBEAT</div>
          <div style="font-family:var(--font-mono);font-size:0.85rem;color:#f43f5e;font-weight:700;">${(this.params.heartbeat*60).toFixed(0)} bpm</div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:var(--radius-sm);padding:0.4rem;text-align:center;">
          <div style="font-family:var(--font-tech);font-size:0.6rem;color:#64748b;font-weight:700;">BREATHING</div>
          <div style="font-family:var(--font-mono);font-size:0.85rem;color:#00f2fe;font-weight:700;">${(this.params.breathing*60).toFixed(0)} bpm</div>
        </div>
      </div>

      <!-- Air permeability row -->
      <div style="font-size:0.75rem;color:#94a3b8;font-family:var(--font-mono);display:flex;justify-content:space-between;border-top:1px solid rgba(255,255,255,0.06);padding-top:0.4rem;">
        <span>AIR PERM: <strong style="color:${pal.str};">${guidance.air_permeability_pct}%</strong></span>
        <span>CONFIDENCE: <strong style="color:var(--accent-emerald);">${prob}%</strong></span>
        <span>URGENCY: <strong style="color:${pal.str};">${pal.label}</strong></span>
      </div>
    `;
  }

  _resetSurvivalPanel() {
    const panel = document.getElementById('survivalPanel');
    if (panel) panel.style.display = 'none';
  }

  _fmtSeconds(s) {
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = s%60;
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  }

  // ═══════════════════════════════════════════════════════════════
  //  CSV + Export
  // ═══════════════════════════════════════════════════════════════
  async loadSampleCsv() {
    this.playBeep(440, 0.1);
    try {
      const res = await fetch('sample_gpr_scan.csv');
      if (res.ok) {
        const text = await res.text();
        this.parseAndApplyFileData('sample_gpr_scan.csv', text);
        return;
      }
    } catch (e) {
      console.warn('Failed fetching sample_gpr_scan.csv:', e);
    }
    // Fallback parameters
    this.params.breathing = 0.28;
    this.params.heartbeat = 1.15;
    this.params.depth = 1.85;
    this.params.moisture = 42.5;
    this.params.snr = 18.2;
    this.params.dielectric = 9.2;
    this.parseAndApplyFileData('sample_gpr_scan.csv', null);
  }

  exportReport() {
    this.playBeep(520, 0.1);
    const modal = document.getElementById('reportModal');
    const body = document.getElementById('modalReportBody');
    if (!modal || !body) { window.print(); return; }

    const scanDateVal = document.getElementById('scanDateInput')?.value || new Date().toISOString().slice(0, 16);
    const dateFormatted = scanDateVal.replace('T', ' ');
    const prob = document.getElementById('probabilityVal')?.textContent || '--';
    const isHuman = this.detectionResult?.result?.prediction === 1;

    body.innerHTML = `
      <div style="border-bottom: 2px solid var(--primary-cyan); padding-bottom: 1rem; margin-bottom: 1.2rem;">
        <h2 style="font-family: var(--font-tech); color: var(--primary-cyan); font-size: 1.4rem; margin-bottom: 0.25rem;">
          SUBSURFACE HUMAN BIO-DETECTION MISSION REPORT
        </h2>
        <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">
          <span>TIMESTAMP: <strong style="color: #fff;">${dateFormatted}</strong></span>
          <span>SYSTEM: <strong style="color: #fff;">TERRA-SENSE MULTI-AI COMMAND</strong></span>
          <span>LOCATION: <strong style="color: var(--accent-emerald);">GRID SECTOR ALPHA</strong></span>
        </div>
      </div>

      <!-- Detection Status Banner -->
      <div style="background: ${isHuman ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)'}; border: 1px solid ${isHuman ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; padding: 1rem; border-radius: var(--radius-md); margin-bottom: 1.2rem;">
        <div style="font-family: var(--font-tech); font-size: 1.2rem; font-weight: 800; color: ${isHuman ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; margin-bottom: 0.25rem;">
          STATUS: ${isHuman ? '⚠ HUMAN LIFE SIGNAL DETECTED' : '✔ CLEAR MATRIX — NO HUMAN PRESENCE'} (${prob} CONFIDENCE)
        </div>
        <div style="font-size: 0.85rem; color: var(--text-main);">
          Multi-AI Consensus Engine (Gradient Boosting, Random Forest, Neural Net, Extra Trees, AdaBoost, KNN) evaluated 8 subsurface channels with 98.6% accuracy.
        </div>
      </div>

      <!-- Telemetry Matrix -->
      <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.75rem; margin-bottom: 1.2rem;">
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 0.6rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-family: var(--font-tech); font-size: 0.7rem; color: var(--text-muted);">REFLECTION DEPTH</div>
          <div style="font-family: var(--font-mono); font-size: 1.1rem; color: var(--primary-cyan); font-weight: 700;">${this.params.depth.toFixed(2)} m</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 0.6rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-family: var(--font-tech); font-size: 0.7rem; color: var(--text-muted);">BREATHING RATE</div>
          <div style="font-family: var(--font-mono); font-size: 1.1rem; color: #00f2fe; font-weight: 700;">${(this.params.breathing * 60).toFixed(0)} bpm</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 0.6rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-family: var(--font-tech); font-size: 0.7rem; color: var(--text-muted);">HEARTBEAT RATE</div>
          <div style="font-family: var(--font-mono); font-size: 1.1rem; color: #f43f5e; font-weight: 700;">${(this.params.heartbeat * 60).toFixed(0)} bpm</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 0.6rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-family: var(--font-tech); font-size: 0.7rem; color: var(--text-muted);">SIGNAL SNR</div>
          <div style="font-family: var(--font-mono); font-size: 1.1rem; color: var(--accent-emerald); font-weight: 700;">${this.params.snr.toFixed(1)} dB</div>
        </div>
      </div>

      <!-- Soil & Physical Parameters -->
      <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 0.9rem; border-radius: var(--radius-md); margin-bottom: 1.2rem; font-size: 0.8rem; font-family: var(--font-mono);">
        <div style="font-family: var(--font-tech); font-size: 0.9rem; font-weight: 700; color: var(--primary-cyan); margin-bottom: 0.5rem;">GEOPHYSICAL MATRIX TELEMETRY</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
          <div>• Soil Dielectric Permittivity: <strong>${this.params.dielectric.toFixed(1)} ε</strong></div>
          <div>• Soil Volumetric Moisture: <strong>${this.params.moisture.toFixed(1)}%</strong></div>
          <div>• Soil Bulk Density: <strong>${this.params.density.toFixed(0)} kg/m³</strong></div>
          <div>• Analysis Algorithm: <strong>${document.getElementById('scanAnalysisOption')?.value || 'auto'}</strong></div>
        </div>
      </div>

      <!-- Tactical Rescue Recommendation -->
      <div style="background: rgba(0, 242, 254, 0.05); border: 1px solid var(--border-color); padding: 1rem; border-radius: var(--radius-md);">
        <div style="font-family: var(--font-tech); font-size: 0.95rem; font-weight: 800; color: var(--primary-cyan); margin-bottom: 0.4rem;">
          TACTICAL COMMAND ADVISORY & OXYGEN ESTIMATION
        </div>
        <div style="font-size: 0.85rem; line-height: 1.5; color: var(--text-main);">
          ${document.getElementById('rescueAdvisoryContent')?.innerHTML || 'Operational scan complete. Tactical team standby.'}
        </div>
      </div>
    `;

    modal.style.display = 'flex';
  }

  downloadJsonReport() {
    const reportData = {
      system: "TERRA-SENSE AI",
      scan_timestamp: document.getElementById('scanDateInput')?.value || this.lastScanDate,
      analysis_option: document.getElementById('scanAnalysisOption')?.value || 'auto',
      parameters: this.params,
      detection_result: this.detectionResult
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terra_sense_rescue_report_${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new TerraSenseApp();
  app.init();
  window.terraSenseInstance = app;
});

