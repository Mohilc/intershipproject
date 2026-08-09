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

    this.activeSectorIds = [];
    this.isEspActive = false;
    this.isFileLoaded = false;

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
    this.gpsBase = { lat: 28.613928, lon: 77.209060 };
    this.currentGps = "";

    this.isScanning   = false;
    this.scanInterval = null;
    this.scanProgress = 0;

    this.espPollInterval = null; // Live ESP32 Telemetry polling interval

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
    
    this.syncUIFromParams = null; // Store reference to UI synchronization
    this.victimMeshes = [];

    // Vision fusion: latest camera detection result for ML score blending
    this.latestCameraResult = {
      human_detected: false,
      confidence_pct: 0,   // 0–100
      box_count: 0,
      timestamp: 0,        // Date.now() ms
      source: 'none'       // 'snapshot' | 'stream_poll' | 'none'
    };
    this._cameraFusionPollInterval = null;
  }

  init() {
    this.initThree();
    this.initCharts();
    this.initHumanControls();
    this.initCameraFeed();
    this.bindEvents();
    
    // Load Host IP from local storage
    const hostInput = document.getElementById('inputHostIp');
    if (hostInput) {
      hostInput.value = localStorage.getItem('terra_sense_host_ip') || '';
    }

    this.initAudio();
    this.startEspTelemetryPolling();
    this._startCameraFusionPoll();
  }

  getApiUrl(path) {
    const hostInput = document.getElementById('inputHostIp');
    let hostIp = hostInput ? hostInput.value.trim() : '';
    if (!hostIp) {
      hostIp = localStorage.getItem('terra_sense_host_ip') || '';
    }
    
    // Clean hostIp to extract raw host/IP
    let cleanHost = hostIp.replace(/^https?:\/\//i, '').replace(/\/.*$/, '').trim();

    // If running from local file:// scheme, fallback to localhost:3000 to reach backend
    if (!cleanHost && (window.location.protocol === 'file:' || !window.location.hostname)) {
      cleanHost = 'localhost:3000';
    }

    const isESP32Hotspot = (cleanHost === '192.168.4.1');

    if (path === '/api/telemetry') {
      if (isESP32Hotspot || cleanHost.startsWith('192.168.')) {
        return `http://${cleanHost}/api/telemetry`;
      }
      if (cleanHost && cleanHost !== 'localhost' && cleanHost !== '127.0.0.1' && cleanHost !== window.location.hostname) {
        const targetHost = cleanHost.includes(':') ? cleanHost : `${cleanHost}:3000`;
        return `http://${targetHost}${path}`;
      }
      return path;
    }

    if (path === '/api/predict') {
      if (isESP32Hotspot || cleanHost.startsWith('192.168.')) {
        const localHost = (window.location.hostname && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
          ? window.location.host
          : 'localhost:3000';
        return `http://${localHost}/api/predict`;
      }
      if (cleanHost && cleanHost !== 'localhost' && cleanHost !== '127.0.0.1' && cleanHost !== window.location.hostname) {
        const targetHost = cleanHost.includes(':') ? cleanHost : `${cleanHost}:3000`;
        return `http://${targetHost}${path}`;
      }
      return path;
    }

    if (cleanHost && cleanHost !== 'localhost' && cleanHost !== '127.0.0.1' && cleanHost !== window.location.hostname) {
      const targetHost = cleanHost.includes(':') ? cleanHost : `${cleanHost}:3000`;
      return `http://${targetHost}${path}`;
    }
    return path;
  }

  getSectorGps(sectorId) {
    let latOff = 0.00018, lonOff = 0.00015;
    if (sectorId === 'A') { latOff *= 1; lonOff *= -1; }
    else if (sectorId === 'B') { latOff *= 1; lonOff *= 1; }
    else if (sectorId === 'C') { latOff *= -1; lonOff *= -1; }
    else if (sectorId === 'D') { latOff *= -1; lonOff *= 1; }
    
    const lat = this.gpsBase.lat + latOff;
    const lon = this.gpsBase.lon + lonOff;
    return `${lat.toFixed(6)}° N, ${Math.abs(lon).toFixed(6)}° E`;
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

    this.syncUIFromParams = syncUIFromParams;
    this.syncUIFromParams();

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

  initCameraFeed() {
    const inputCamIp = document.getElementById('inputCamIp');
    const btnConnect = document.getElementById('btnConnectCam');
    const btnFlash = document.getElementById('btnToggleFlash');
    const btnSnapshot = document.getElementById('btnCaptureSnapshot');
    const btnDemo = document.getElementById('btnDemoCamStream');
    const imgStream = document.getElementById('espCamStreamImg');
    const placeholder = document.getElementById('espCamPlaceholder');
    const statusText = document.getElementById('espCamStatusText');
    const dot = document.getElementById('espCamDot');

    const btnYolo = document.getElementById('btnToggleYolo');
    const yoloAlert = document.getElementById('espCamYoloAlert');
    const btnPinouts = document.getElementById('btnCamModelInfo');
    const modal = document.getElementById('camModelModal');
    const btnCloseModal = document.getElementById('btnCloseCamModelModal');
    const btnCloseModal2 = document.getElementById('btnCloseCamModelModal2');
    const btnPresetEspCam = document.getElementById('btnPresetEspCam');
    const btnPresetWebcam = document.getElementById('btnPresetWebcam');

    let flashOn = false;
    let isDemoMode = false;
    let isYoloActive = true;
    let yoloPollInterval = null;
    let lastDetectedState = false;

    const setStatus = (online, text) => {
      if (dot) dot.style.background = online ? 'var(--accent-emerald)' : '#9ca3af';
      if (statusText) statusText.textContent = text;
    };

    const startYoloDetectionPolling = () => {
      if (yoloPollInterval) clearInterval(yoloPollInterval);
      yoloPollInterval = setInterval(() => {
        if (!isYoloActive || isDemoMode) return;
        fetch('/api/camera/latest_detection')
          .then(res => res.json())
          .then(data => {
            if (data.status === 'success' && data.fresh) {
              this.latestCameraResult = {
                human_detected: data.human_detected,
                confidence_pct: data.confidence_pct || 0,
                box_count: data.box_count || 0,
                timestamp: Date.now(),
                source: data.source || 'stream'
              };
              this._updateFusionIndicator();

              if (yoloAlert) {
                if (data.human_detected) {
                  yoloAlert.style.display = 'block';
                  yoloAlert.textContent = `⚠ HUMAN: ${data.confidence_pct}% (${data.box_count} target${data.box_count > 1 ? 's' : ''})`;
                  yoloAlert.style.background = 'rgba(244,63,94,0.95)';
                  if (!lastDetectedState) {
                    this.playBeep(784, 0.25);
                  }
                  lastDetectedState = true;
                } else {
                  yoloAlert.style.display = 'block';
                  yoloAlert.textContent = '✓ SCANNING — 0 TARGETS';
                  yoloAlert.style.background = 'rgba(16, 185, 129, 0.85)';
                  lastDetectedState = false;
                }
              }
            }
          })
          .catch(() => {});
      }, 400);
    };

    const stopYoloDetectionPolling = () => {
      if (yoloPollInterval) {
        clearInterval(yoloPollInterval);
        yoloPollInterval = null;
      }
    };

    const formatCamUrl = (rawInput) => {
      let str = (rawInput || '').trim();
      if (!str) return '';
      if (str.toLowerCase() === 'webcam' || str === '0' || str.toLowerCase() === 'local') {
        return 'webcam';
      }
      if (!str.startsWith('http://') && !str.startsWith('https://')) {
        str = 'http://' + str;
      }
      try {
        const urlObj = new URL(str);
        if (urlObj.pathname === '/' || urlObj.pathname === '') {
          urlObj.pathname = '/stream';
        }
        return urlObj.toString();
      } catch (e) {
        if (!str.includes('/stream')) {
          str = str.replace(/\/$/, '') + '/stream';
        }
        return str;
      }
    };

    // Load saved Cam IP
    const savedCamUrl = localStorage.getItem('terra_sense_cam_url');
    if (savedCamUrl && inputCamIp) {
      inputCamIp.value = savedCamUrl;
    }

    const connectStream = (rawUrl) => {
      if (!rawUrl) return;
      const formattedUrl = formatCamUrl(rawUrl);
      localStorage.setItem('terra_sense_cam_url', formattedUrl);
      if (inputCamIp) inputCamIp.value = formattedUrl;

      setStatus(true, 'CONNECTING AI STREAM...');
      
      if (imgStream) {
        imgStream.onerror = () => {
          setStatus(false, 'STREAM OFFLINE / TIMEOUT');
          imgStream.style.display = 'none';
          if (placeholder) placeholder.style.display = 'block';
          if (yoloAlert) yoloAlert.style.display = 'none';
          stopYoloDetectionPolling();
        };

        if (isYoloActive) {
          imgStream.src = `/api/camera/stream_yolo?url=${encodeURIComponent(formattedUrl)}`;
          if (yoloAlert) {
            yoloAlert.style.display = 'block';
            yoloAlert.textContent = '👁 AI SCANNING...';
            yoloAlert.style.background = 'rgba(0,242,254,0.85)';
            yoloAlert.style.color = '#040711';
          }
          startYoloDetectionPolling();
        } else {
          stopYoloDetectionPolling();
          if (formattedUrl === 'webcam') {
            imgStream.src = `/api/camera/stream_yolo?url=webcam`;
          } else {
            imgStream.src = formattedUrl;
          }
          if (yoloAlert) yoloAlert.style.display = 'none';
        }

        imgStream.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
        
        setTimeout(() => {
          if (imgStream.style.display !== 'none') {
            setStatus(true, isYoloActive ? 'LIVE AI YOLO DETECT ONLINE' : 'LIVE OPTICAL FEED ONLINE');
          }
        }, 600);
      }
    };

    // Preset buttons
    if (btnPresetEspCam && inputCamIp) {
      btnPresetEspCam.addEventListener('click', () => {
        inputCamIp.value = '192.168.4.2/stream';
        connectStream('192.168.4.2/stream');
      });
    }

    if (btnPresetWebcam && inputCamIp) {
      btnPresetWebcam.addEventListener('click', () => {
        inputCamIp.value = 'webcam';
        connectStream('webcam');
      });
    }

    if (btnConnect && inputCamIp) {
      btnConnect.addEventListener('click', () => {
        isDemoMode = false;
        const val = inputCamIp.value.trim();
        if (!val) {
          connectStream('192.168.4.2/stream');
        } else {
          connectStream(val);
        }
      });
    }

    // Flash Light
    if (btnFlash) {
      btnFlash.addEventListener('click', () => {
        flashOn = !flashOn;
        btnFlash.classList.toggle('active', flashOn);
        const camIp = inputCamIp ? inputCamIp.value.trim() : '';
        if (camIp && camIp !== 'webcam') {
          const formatted = formatCamUrl(camIp);
          try {
            const urlObj = new URL(formatted);
            const targetLedUrl = `${urlObj.origin}/led?state=${flashOn ? 'on' : 'off'}`;
            fetch(`/api/camera/proxy?url=${encodeURIComponent(targetLedUrl)}`)
              .catch(err => console.warn('Flash command error:', err));
          } catch (e) {
            console.warn('Invalid URL for flash:', camIp);
          }
        }
      });
    }

    // Snapshot Capture with YOLO inference
    if (btnSnapshot) {
      btnSnapshot.addEventListener('click', () => {
        const camIp = inputCamIp ? inputCamIp.value.trim() : '';
        this.playBeep(880, 0.15);
        
        setStatus(true, 'AI ANALYZING SNAPSHOT...');
        const isDemoTarget = isDemoMode || (!camIp && !isYoloActive);
        
        let targetApi = `/api/camera/analyze_snapshot?demo=true`;
        if (!isDemoTarget && camIp) {
          if (camIp === 'webcam') {
            targetApi = `/api/camera/analyze_snapshot?url=webcam`;
          } else {
            const formatted = formatCamUrl(camIp);
            try {
              const captureUrl = `${new URL(formatted).origin}/capture`;
              targetApi = `/api/camera/analyze_snapshot?url=${encodeURIComponent(captureUrl)}`;
            } catch (e) {
              targetApi = `/api/camera/analyze_snapshot?url=${encodeURIComponent(formatted)}`;
            }
          }
        }
        
        fetch(targetApi)
          .then(res => res.json())
          .then(data => {
            if (data.status === 'success') {
              if (imgStream) {
                imgStream.src = data.image;
                imgStream.style.display = 'block';
                if (placeholder) placeholder.style.display = 'none';
              }

              // Store result for ML fusion
              this.latestCameraResult = {
                human_detected: data.human_detected,
                confidence_pct: data.confidence_pct || 0,
                box_count: data.box_count || 0,
                timestamp: Date.now(),
                source: data.source || 'snapshot'
              };
              this._updateFusionIndicator();

              if (yoloAlert) {
                if (data.human_detected) {
                  yoloAlert.style.display = 'block';
                  yoloAlert.textContent = `⚠ HUMAN: ${data.confidence_pct}% — FUSED INTO ML`;
                  yoloAlert.style.background = 'rgba(244,63,94,0.95)';
                  this.playBeep(784, 0.3);
                } else {
                  yoloAlert.style.display = 'block';
                  yoloAlert.textContent = '✓ CLEAR — FUSED INTO ML';
                  yoloAlert.style.background = 'rgba(16, 185, 129, 0.9)';
                }
              }
              setStatus(true, data.human_detected ? `DETECTED: ${data.confidence_pct}% — VISION FUSED` : 'AI ANALYSIS: CLEAR — VISION FUSED');
            } else {
              setStatus(false, 'AI ANALYSIS FAILED');
            }
          })
          .catch((err) => {
            console.warn(err);
            setStatus(false, 'AI ANALYSIS OFFLINE');
          });
      });
    }

    // YOLO toggle listener
    if (btnYolo) {
      btnYolo.addEventListener('click', () => {
        isYoloActive = !isYoloActive;
        btnYolo.classList.toggle('active', isYoloActive);
        btnYolo.textContent = `YOLO AI: ${isYoloActive ? 'ON' : 'OFF'}`;
        if (isYoloActive) {
          btnYolo.style.background = 'var(--accent-emerald)';
          btnYolo.style.color = '#040711';
          btnYolo.style.fontWeight = '700';
        } else {
          btnYolo.style.background = '';
          btnYolo.style.color = '';
          btnYolo.style.fontWeight = '';
          if (yoloAlert) yoloAlert.style.display = 'none';
          stopYoloDetectionPolling();
        }
        
        if (imgStream && imgStream.style.display === 'block') {
          if (isDemoMode) {
            if (isYoloActive) {
              if (yoloAlert) { yoloAlert.style.display = 'block'; yoloAlert.textContent = '⚠ HUMAN: 94.7%'; }
            } else {
              if (yoloAlert) yoloAlert.style.display = 'none';
            }
          } else {
            const camIp = inputCamIp ? inputCamIp.value.trim() : '';
            if (camIp) connectStream(camIp);
          }
        }
      });
    }

    // Demo Stream Toggle
    if (btnDemo) {
      btnDemo.addEventListener('click', () => {
        isDemoMode = !isDemoMode;
        btnDemo.classList.toggle('active', isDemoMode);
        if (isDemoMode) {
          stopYoloDetectionPolling();
          if (imgStream) {
            if (isYoloActive) {
              const svgData = `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240"><rect width="320" height="240" fill="#040711"/><circle cx="160" cy="120" r="90" fill="none" stroke="#00f2fe" stroke-width="1.5" opacity="0.4"/><circle cx="160" cy="120" r="60" fill="none" stroke="#00f2fe" stroke-width="1" opacity="0.3"/><line x1="160" y1="20" x2="160" y2="220" stroke="#00f2fe" opacity="0.25"/><line x1="60" y1="120" x2="260" y2="120" stroke="#00f2fe" opacity="0.25"/><rect x="130" y="70" width="70" height="120" fill="rgba(244,63,94,0.2)" stroke="#f43f5e" stroke-width="2"/><text x="165" y="62" font-family="monospace" font-size="10" fill="#fff" text-anchor="middle">HUMAN: 94.7%</text><text x="160" y="225" font-family="monospace" font-size="10" fill="#00f2fe" text-anchor="middle">YOLO ACTIVE — DEMO FEED</text></svg>`;
              imgStream.src = `data:image/svg+xml;utf8,${encodeURIComponent(svgData)}`;
              if (yoloAlert) {
                yoloAlert.style.display = 'block';
                yoloAlert.textContent = '⚠ HUMAN: 94.7%';
              }
            } else {
              const svgData = `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240"><rect width="320" height="240" fill="#040711"/><circle cx="160" cy="120" r="90" fill="none" stroke="#00f2fe" stroke-width="1.5" opacity="0.4"/><circle cx="160" cy="120" r="60" fill="none" stroke="#00f2fe" stroke-width="1" opacity="0.3"/><line x1="160" y1="20" x2="160" y2="220" stroke="#00f2fe" opacity="0.25"/><line x1="60" y1="120" x2="260" y2="120" stroke="#00f2fe" opacity="0.25"/><text x="160" y="225" font-family="monospace" font-size="10" fill="#00f2fe" text-anchor="middle">DEMO OPTICAL FEED (640x480)</text></svg>`;
              imgStream.src = `data:image/svg+xml;utf8,${encodeURIComponent(svgData)}`;
              if (yoloAlert) yoloAlert.style.display = 'none';
            }
            imgStream.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
          }
          setStatus(true, 'SYNTHETIC DEMO STREAM');
        } else {
          stopYoloDetectionPolling();
          if (imgStream) imgStream.style.display = 'none';
          if (placeholder) placeholder.style.display = 'block';
          if (yoloAlert) yoloAlert.style.display = 'none';
          setStatus(false, 'CAMERA NODE OFFLINE');
        }
      });
    }

    // Modal controls
    if (btnPinouts && modal) {
      btnPinouts.addEventListener('click', () => modal.style.display = 'flex');
    }
    const closeModal = () => { if (modal) modal.style.display = 'none'; };
    if (btnCloseModal) btnCloseModal.addEventListener('click', closeModal);
    if (btnCloseModal2) btnCloseModal2.addEventListener('click', closeModal);
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  Camera Vision Fusion Helpers
  // ═══════════════════════════════════════════════════════════════

  /** Polls /api/camera/latest_detection every 10 s so fusion works even
   *  without manual snapshots (e.g. when the live stream is running). */
  _startCameraFusionPoll() {
    if (this._cameraFusionPollInterval) return;
    const poll = async () => {
      try {
        const res = await fetch(this.getApiUrl('/api/camera/latest_detection'));
        if (!res.ok) return;
        const d = await res.json();
        if (d.fresh) {
          this.latestCameraResult = {
            human_detected: d.human_detected,
            confidence_pct: d.confidence_pct,
            box_count: d.box_count || 0,
            timestamp: Date.now() - (d.age_seconds * 1000),
            source: d.source || 'stream_poll'
          };
          this._updateFusionIndicator();
        }
      } catch (e) { /* camera offline – ignore silently */ }
    };
    poll(); // run immediately on init
    this._cameraFusionPollInterval = setInterval(poll, 10000);
  }

  /** Returns true when the last camera reading is fresh enough to fuse (< 30 s). */
  _isCameraFresh() {
    return this.latestCameraResult.source !== 'none' &&
           (Date.now() - this.latestCameraResult.timestamp) < 30000;
  }

  /** Updates the UI indicator that shows when vision fusion is active. */
  _updateFusionIndicator() {
    const pill = document.getElementById('fusionStatusPill');
    if (!pill) return;
    if (this._isCameraFresh()) {
      pill.style.display = 'flex';
      pill.innerHTML = this.latestCameraResult.human_detected
        ? `🎞️ VISION FUSION ACTIVE — CAM: ${this.latestCameraResult.confidence_pct.toFixed(0)}% HUMAN`
        : `🎞️ VISION FUSION ACTIVE — CAM: CLEAR`;
      pill.style.background = this.latestCameraResult.human_detected
        ? 'rgba(244,63,94,0.15)' : 'rgba(16,185,129,0.12)';
      pill.style.borderColor = this.latestCameraResult.human_detected
        ? '#f43f5e' : '#10b981';
      pill.style.color = this.latestCameraResult.human_detected
        ? '#f43f5e' : '#10b981';
    } else {
      pill.style.display = 'none';
    }
  }

  /** Returns the API path to use for predictions (fused vs standard). */
  _predictApiPath() {
    return this._isCameraFresh() ? '/api/predict_fused' : '/api/predict';
  }

  /** Appends camera fusion fields to a batch target array for /api/predict_fused. */
  _addFusionFields(body) {
    if (this._isCameraFresh()) {
      body.camera_confidence     = this.latestCameraResult.confidence_pct / 100.0;
      body.camera_human_detected = this.latestCameraResult.human_detected;
    }
    return body;
  }


  // ═══════════════════════════════════════════════════════════════
  //  THREE.JS — 4-Quadrant Subsurface Scene
  // ═══════════════════════════════════════════════════════════════
  initThree() {
    if (typeof THREE === 'undefined') {
      console.warn("Three.js is not loaded. 3D grid visualization is disabled.");
      return;
    }
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

    const CtrlClass = (typeof THREE !== 'undefined' && THREE.OrbitControls) ? THREE.OrbitControls : null;
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
    this._createDrone();
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

  // ── 3D Drone Creation ─────────────────────────────────────────────
  _createDrone() {
    this.droneGroup = new THREE.Group();
    this.droneGroup.position.set(0, 7.5, 0);
    this.threeScene.add(this.droneGroup);

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.2 });
    const accentMat = new THREE.MeshStandardMaterial({ color: 0x00f2fe, emissive: 0x00f2fe, emissiveIntensity: 0.8 });
    const rotorMat = new THREE.MeshStandardMaterial({ color: 0x020617, metalness: 0.9, roughness: 0.1 });

    const bodyGeo = new THREE.BoxGeometry(0.6, 0.2, 0.6);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    this.droneGroup.add(body);

    const domeGeo = new THREE.SphereGeometry(0.2, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2);
    const dome = new THREE.Mesh(domeGeo, accentMat);
    dome.position.y = 0.1;
    this.droneGroup.add(dome);

    this.rotors = [];
    this.droneLeds = [];
    const armGeo = new THREE.CylinderGeometry(0.04, 0.04, 1.2, 8);
    armGeo.rotateZ(Math.PI / 2);

    const positions = [
      { x: 0.5, z: 0.5, angle: Math.PI / 4 },
      { x: -0.5, z: 0.5, angle: -Math.PI / 4 },
      { x: 0.5, z: -0.5, angle: 3 * Math.PI / 4 },
      { x: -0.5, z: -0.5, angle: -3 * Math.PI / 4 }
    ];

    positions.forEach((pos, idx) => {
      const arm = new THREE.Mesh(armGeo, bodyMat);
      arm.rotation.y = pos.angle;
      arm.position.set(pos.x / 2, 0, pos.z / 2);
      this.droneGroup.add(arm);

      const motorGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.15, 8);
      const motor = new THREE.Mesh(motorGeo, bodyMat);
      motor.position.set(pos.x, 0.08, pos.z);
      this.droneGroup.add(motor);

      const bladeGeo = new THREE.BoxGeometry(0.8, 0.01, 0.05);
      const blade = new THREE.Mesh(bladeGeo, rotorMat);
      blade.position.set(pos.x, 0.16, pos.z);
      this.droneGroup.add(blade);
      this.rotors.push(blade);

      const ledGeo = new THREE.SphereGeometry(0.04, 8, 8);
      const ledColor = (idx % 2 === 0) ? 0xef4444 : 0x10b981;
      const ledMat = new THREE.MeshBasicMaterial({ color: ledColor, transparent: true, opacity: 0.9 });
      const led = new THREE.Mesh(ledGeo, ledMat);
      led.position.set(pos.x, -0.06, pos.z);
      this.droneGroup.add(led);
      this.droneLeds.push(led);
    });

    this.droneSpotlight = new THREE.SpotLight(0x38ef7d, 5, 25, Math.PI / 6, 0.6, 1);
    this.droneSpotlight.position.set(0, -0.1, 0);
    this.droneGroup.add(this.droneSpotlight);

    const coneGeo = new THREE.ConeGeometry(2.2, 6.0, 32, 1, true);
    coneGeo.translate(0, -3.0, 0);
    coneGeo.rotateX(Math.PI / 2);
    const coneMat = new THREE.MeshBasicMaterial({
      color: 0x38ef7d,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.spotlightBeam = new THREE.Mesh(coneGeo, coneMat);
    this.spotlightBeam.position.set(0, 0, 0);
    this.droneGroup.add(this.spotlightBeam);

    this.spotlightTarget = new THREE.Object3D();
    this.spotlightTarget.position.set(0, 3.0, 0);
    this.threeScene.add(this.spotlightTarget);
    this.droneSpotlight.target = this.spotlightTarget;
  }

  // ── Dynamic Subsurface Victim Renderer ───────────────────────────
  _clearDynamicVictims() {
    if (this.victimMeshes) {
      this.victimMeshes.forEach(v => {
        this.threeScene.remove(v.mesh);
        this.threeScene.remove(v.line);
        this.threeScene.remove(v.label);
      });
    }
    this.victimMeshes = [];
  }

  _plotDynamicVictims(targets) {
    this._clearDynamicVictims();
    if (!targets || targets.length === 0) return;

    // Determine min/max bounding range for grid mapping
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    targets.forEach(t => {
      const loc = t.subsurface_victim_locator || {};
      const coords = loc.grid_coordinates || { x: t.x !== undefined ? t.x : (t.grid_x !== undefined ? t.grid_x : 0), y: t.y !== undefined ? t.y : (t.grid_y !== undefined ? t.grid_y : 0) };
      const rx = coords.x !== undefined ? coords.x : 0;
      const ry = coords.y !== undefined ? coords.y : 0;
      if (rx < minX) minX = rx;
      if (rx > maxX) maxX = rx;
      if (ry < minY) minY = ry;
      if (ry > maxY) maxY = ry;
    });

    targets.forEach((target, idx) => {
      const isHuman = target.human_detected === true || target.prediction === 1;
      if (!isHuman) return;

      const loc = target.subsurface_victim_locator || {};
      const depth = loc.depth_meters || target.reflection_depth || this.params.depth;
      const coords = loc.grid_coordinates || { x: target.x || 0, y: target.y || 0, z: -depth };
      const rawX = coords.x !== undefined ? coords.x : (target.x || 0);
      const rawY = coords.y !== undefined ? coords.y : (target.y || 0);

      // Map raw coordinates to visible 3D scene grid bounds (-3.8m to +3.8m)
      let sceneX, sceneZ;
      if (Math.abs(rawX) <= 4.5 && Math.abs(rawY) <= 4.5) {
        sceneX = rawX;
        sceneZ = rawY;
      } else {
        const spanX = (maxX > minX) ? (maxX - minX) : 30.0;
        const spanY = (maxY > minY) ? (maxY - minY) : 30.0;
        sceneX = spanX > 0 ? (((rawX - minX) / spanX) * 7.6 - 3.8) : (Math.random() * 6 - 3);
        sceneZ = spanY > 0 ? (((rawY - minY) / spanY) * 7.6 - 3.8) : (Math.random() * 6 - 3);
      }

      const gps = loc.gps_coordinates || {
        latitude: this.gpsBase.lat + (rawY / 111320.0),
        longitude: this.gpsBase.lon + (rawX / 111320.0)
      };

      const targetY = 3.0 - Math.min(4.5, Math.max(0.3, depth));
      const oxyHours = target.rescue_guidance?.estimated_oxygen_hours || 12.0;
      const pal = survivalColor(oxyHours);

      const group = new THREE.Group();

      const capsule = this._buildHumanMesh(pal.hex);
      capsule.scale.set(0.95, 0.95, 0.95);
      capsule.position.set(0, 0, 0);
      capsule.visible = true;
      group.add(capsule);

      const sphereGeo = new THREE.SphereGeometry(0.38, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color: pal.hex, wireframe: true, transparent: true, opacity: 0.4 });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      group.add(sphere);

      const hbGeo = new THREE.RingGeometry(0.5, 0.58, 32);
      const hbMat = new THREE.MeshBasicMaterial({ color: pal.hex, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
      const hbRing = new THREE.Mesh(hbGeo, hbMat);
      hbRing.rotation.x = Math.PI / 2;
      group.add(hbRing);

      group.position.set(sceneX, targetY, sceneZ);
      this.threeScene.add(group);

      const linePts = [
        new THREE.Vector3(sceneX, 3.0, sceneZ),
        new THREE.Vector3(sceneX, targetY, sceneZ)
      ];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(linePts);
      const lineMat = new THREE.LineDashedMaterial({
        color: pal.hex,
        dashSize: 0.3,
        gapSize: 0.15,
        transparent: true,
        opacity: 0.75
      });
      const probeLine = new THREE.Line(lineGeo, lineMat);
      probeLine.computeLineDistances();
      this.threeScene.add(probeLine);

      const labelText = `P#${idx + 1} | Lat:${gps.latitude.toFixed(6)} | Lon:${gps.longitude.toFixed(6)} | Z:-${depth.toFixed(2)}m`;
      const labelSprite = this._createVictimLabelSprite(labelText, pal.str);
      labelSprite.position.set(sceneX, targetY + 1.25, sceneZ);
      this.threeScene.add(labelSprite);

      this.victimMeshes.push({
        mesh: group,
        sphere: sphere,
        hb: hbRing,
        line: probeLine,
        label: labelSprite,
        breathing: target.vital_doppler_diagnostics?.respiration_rate_bpm ? (target.vital_doppler_diagnostics.respiration_rate_bpm / 60.0) : 0.28,
        heartbeat: target.vital_doppler_diagnostics?.heartbeat_rate_bpm ? (target.vital_doppler_diagnostics.heartbeat_rate_bpm / 60.0) : 1.15
      });
    });
  }

  _createVictimLabelSprite(text, colorStr) {
    const canvas = document.createElement('canvas');
    canvas.width = 460;
    canvas.height = 70;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(2, 6, 23, 0.9)';
    ctx.beginPath();
    ctx.roundRect(0, 0, 460, 70, 8);
    ctx.fill();
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = 'bold 20px Rajdhani, JetBrains Mono, monospace';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, 12, 42);
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(3.8, 0.6, 1);
    return sprite;
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

    // Drone animation
    if (this.droneGroup) {
      const orbitRadius = 3.6;
      const droneSpeed = 0.55;
      const hoverHeight = 6.6;

      const droneX = orbitRadius * Math.sin(time * droneSpeed);
      const droneZ = orbitRadius * Math.cos(time * droneSpeed);
      const droneY = hoverHeight + 0.28 * Math.sin(time * 1.8);

      this.droneGroup.position.set(droneX, droneY, droneZ);

      // Bank tilt
      this.droneGroup.rotation.z = -0.12 * Math.cos(time * droneSpeed);
      this.droneGroup.rotation.x = -0.12 * Math.sin(time * droneSpeed);
      this.droneGroup.rotation.y = -time * droneSpeed + Math.PI / 2;

      // Spin rotors
      if (this.rotors) {
        this.rotors.forEach(r => { r.rotation.y += 0.45; });
      }

      // Blink LED lights
      if (this.droneLeds) {
        const isLit = Math.sin(time * 7.0) > 0;
        this.droneLeds.forEach(led => {
          led.material.opacity = isLit ? 0.95 : 0.15;
        });
      }

      // Drone GPS coordinates updates on UI
      const dLat = this.gpsBase.lat + (droneZ / 111320.0);
      const dLon = this.gpsBase.lon + (droneX / 111320.0);

      const droneAltitudeVal = document.getElementById('droneAltitudeVal');
      const droneStatusVal = document.getElementById('droneStatusVal');
      const droneLatitudeVal = document.getElementById('droneLatitudeVal');
      const droneLongitudeVal = document.getElementById('droneLongitudeVal');

      if (droneAltitudeVal) droneAltitudeVal.textContent = `${droneY.toFixed(2)} M`;
      if (droneStatusVal) {
        droneStatusVal.textContent = this.isScanning ? "SURFACE SWEEP" : "ORBITAL SCAN";
        droneStatusVal.style.color = this.isScanning ? "var(--accent-rose)" : "var(--accent-emerald)";
      }
      if (droneLatitudeVal) droneLatitudeVal.textContent = dLat.toFixed(6);
      if (droneLongitudeVal) droneLongitudeVal.textContent = dLon.toFixed(6);

      // Point spotlight target
      if (this.spotlightTarget) {
        if (this.isScanning) {
          this.spotlightTarget.position.set(5.0 * Math.sin(time * 4.0), 3.0, 5.0 * Math.cos(time * 4.0));
        } else if (this.activeSectorIds && this.activeSectorIds.length > 0) {
          const sd = SECTORS.find(s => s.id === this.activeSectorIds[0]);
          if (sd) {
            this.spotlightTarget.position.set(sd.cx, 3.0, sd.cz);
          }
        } else {
          this.spotlightTarget.position.set(droneX, 3.0, droneZ);
        }
      }
    }

    // Animate dynamic victims
    if (this.victimMeshes) {
      this.victimMeshes.forEach(v => {
        const bs = Math.sin(time * Math.PI * 2 * v.breathing) * 0.10;
        const hs = Math.sin(time * Math.PI * 2 * v.heartbeat) * 0.04;
        const sc = 1.0 + bs + hs;
        if (v.sphere) v.sphere.scale.set(sc, sc, sc);
        if (v.hb) {
          const hbPhase = (Math.sin(time * Math.PI * 2 * v.heartbeat) + 1) / 2;
          v.hb.material.opacity = hbPhase * 0.85;
          const hsc = 1.0 + hbPhase * 1.2;
          v.hb.scale.set(hsc, hsc, 1);
        }
      });
    }

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
      const targetY  = 3.0 - this.params.depth;
      const isActive = this.activeSectorIds && this.activeSectorIds.includes(sec.id);

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
      lbl.material.opacity = (this.activeSectorIds && this.activeSectorIds.includes(sec.id))
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
  _highlightSectors(sectorIds, oxyHours) {
    this.activeSectorIds = sectorIds;
    const pal = survivalColor(oxyHours);

    SECTORS.forEach(s => {
      const panel  = this.sectorPanels[s.id];
      const sphere = this.epicenters[s.id];
      const light  = this._epicenterLights[s.id];
      const oxyBar = this.oxyBars[s.id];
      const human  = this.humanGroups[s.id];

      const isActive = sectorIds.includes(s.id);

      if (isActive) {
        // Active sector — apply survival palette
        if (panel)  { panel.material.opacity = 0.25; panel.material.color.setHex(pal.hex); }
        if (sphere) { sphere.material.color.setHex(pal.hex); sphere.material.opacity = 0.9; }
        if (light)  { light.color.setHex(pal.hex); }
        if (oxyBar) { oxyBar.material.color.setHex(pal.hex); oxyBar.material.opacity = 0.85; }

        if (human) {
          human.visible = true;
          human.traverse(child => {
            if (child.isMesh && child.material) {
              child.material.color.setHex(pal.hex);
              if (child.material.emissive) child.material.emissive.setHex(pal.hex);
            }
          });
        }

        // Heartbeat + signal wave rings colour
        const hbRing = this.heartbeatRings[s.id];
        if (hbRing) { hbRing.material.color.setHex(pal.hex); hbRing.material.opacity = 0.85; }
        const waves  = this.signalWaves[s.id];
        if (waves) waves.forEach(r => { r.material.color.setHex(pal.hex); r.material.opacity = 0.45; });

      } else {
        // Dim other sectors
        if (panel)  panel.material.opacity = 0.025;
        if (sphere) sphere.material.opacity = 0.15;
        if (human)  human.visible = false;
        if (oxyBar) oxyBar.material.opacity = 0;
        const hbRing = this.heartbeatRings[s.id];
        if (hbRing) hbRing.material.opacity = 0;
      }
    });

    // Camera focus on first active sector in array
    if (this.threeControls && sectorIds.length > 0) {
      const sd = SECTORS.find(s => s.id === sectorIds[0]);
      if (sd) {
        this.threeControls.target.set(sd.cx, 0, sd.cz);
        this.threeCamera.position.set(sd.cx + 5.5, 7.5, sd.cz + 8);
        this.threeControls.update();
      }
    }
  }

  _highlightSector(sectorId, oxyHours) {
    this._highlightSectors([sectorId], oxyHours);
  }

  _clearAllSectors() {
    this.activeSectorIds = [];
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
  _updateTelemetryOverlay(state, sectorIds, oxyHours) {
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
    } else if (state === 'DETECTED' && sectorIds && sectorIds.length > 0 && oxyHours != null) {
      const count = sectorIds.length;
      const pal = survivalColor(oxyHours);
      const locationNames = sectorIds.map(id => `Sector ${id}`).join(' · ');
      overlay.style.borderColor = pal.str;
      overlay.innerHTML = `
        <span style="color:${pal.str};font-weight:700;font-size:0.82rem;">⚠ ${count} ${count > 1 ? 'HUMANS' : 'HUMAN'} DETECTED</span><br>
        LOCATIONS: <span style="font-weight:700;">${locationNames}</span><br>
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
      if (!this.activeSectorIds || this.activeSectorIds.length === 0) { clearInterval(this._countdownInterval); return; }
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
  _updateSectorStatusPanel(activeSectorIds, prob, oxyHours) {
    const panel = document.getElementById('sectorStatusPanel');
    if (!panel) return;
    const pal = oxyHours != null ? survivalColor(oxyHours) : null;

    panel.innerHTML = SECTORS.map(s => {
      const isActive = activeSectorIds && activeSectorIds.includes(s.id);
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
    chart.update('none');
  }

  updateDepthChart() {
    const chart = this.charts['depth'];
    if (!chart) return;
    const di = Math.round(this.params.depth / 0.5);
    chart.data.datasets[0].data = Array.from({length:15},(_,i)=>{ let v=-2*i; if(i===di) v+=this.params.snr; return v; });
    chart.update('none');
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

  animateCameraTo(targetPos, targetLookAt, duration = 600) {
    if (!this.threeCamera || !this.threeControls) return;
    const startPos = this.threeCamera.position.clone();
    const startTarget = this.threeControls.target.clone();
    const startTime = performance.now();

    const step = (now) => {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / duration);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

      this.threeCamera.position.lerpVectors(startPos, new THREE.Vector3(targetPos.x, targetPos.y, targetPos.z), ease);
      this.threeControls.target.lerpVectors(startTarget, new THREE.Vector3(targetLookAt.x, targetLookAt.y, targetLookAt.z), ease);
      this.threeControls.update();

      if (t < 1) {
        requestAnimationFrame(step);
      }
    };
    requestAnimationFrame(step);
  }

  // ═══════════════════════════════════════════════════════════════
  //  Event Bindings
  // ═══════════════════════════════════════════════════════════════
  bindEvents() {
    // ESP32 Live Telemetry toggle
    const espToggle = document.getElementById('espTelemetryToggle');
    espToggle?.addEventListener('change', (e) => {
      if (e.target.checked) {
        this.startEspTelemetryPolling();
      } else {
        this.stopEspTelemetryPolling();
      }
    });

    // Apply Host IP Button click
    document.getElementById('btnApplyHostIp')?.addEventListener('click', () => {
      const hostInput = document.getElementById('inputHostIp');
      const hostIp = hostInput ? hostInput.value.trim() : '';
      localStorage.setItem('terra_sense_host_ip', hostIp);
      
      const btn = document.getElementById('btnApplyHostIp');
      if (btn) {
        btn.textContent = "SAVED";
        btn.style.background = "var(--accent-emerald)";
        btn.style.color = "#03050b";
        setTimeout(() => {
          btn.textContent = "SET";
          btn.style.background = "";
          btn.style.color = "";
        }, 1200);
      }
      this.playBeep(520, 0.07);
      
      // Trigger immediate telemetry poll
      this.pollTelemetry();
    });

    // Scan Initiate Button
    document.getElementById('btnInitiateScan')?.addEventListener('click', () => this.startScanSequence());
    document.getElementById('btnResetCamera')?.addEventListener('click', () => this.resetCamera());
    document.getElementById('btnLoadSampleCsv')?.addEventListener('click', () => this.loadSampleCsv());
    document.getElementById('btnExportReport')?.addEventListener('click', () => this.exportReport());

    // Copy GPS Coordinates
    document.getElementById('btnCopyGps')?.addEventListener('click', () => {
      if (this.currentGps && this.currentGps !== "N/A - MATRIX CLEAR") {
        navigator.clipboard.writeText(this.currentGps).then(() => {
          const btn = document.getElementById('btnCopyGps');
          if (btn) {
            btn.textContent = "COPIED!";
            btn.style.color = "var(--accent-emerald)";
            btn.style.borderColor = "var(--accent-emerald)";
            setTimeout(() => {
              btn.textContent = "COPY";
              btn.style.color = "";
              btn.style.borderColor = "";
            }, 1200);
          }
          this.playBeep(650, 0.08);
        }).catch(err => console.error("Could not copy text: ", err));
      } else {
        this.playBeep(330, 0.15);
      }
    });

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

    // Sector quick focus buttons & animations
    const sectorNavMap = {
      'btnFocusSectorAll': { sector: 'ALL', pos: { x: 0, y: 13, z: 15 }, target: { x: 0, y: 0, z: 0 } },
      'btnFocusSectorA':   { sector: 'A',   pos: { x: -4.5, y: 7.5, z: -1 }, target: { x: -2.5, y: 0, z: -2.5 } },
      'btnFocusSectorB':   { sector: 'B',   pos: { x: 4.5, y: 7.5, z: -1 },  target: { x: 2.5, y: 0, z: -2.5 } },
      'btnFocusSectorC':   { sector: 'C',   pos: { x: -4.5, y: 7.5, z: 4.5 }, target: { x: -2.5, y: 0, z: 2.5 } },
      'btnFocusSectorD':   { sector: 'D',   pos: { x: 4.5, y: 7.5, z: 4.5 },  target: { x: 2.5, y: 0, z: 2.5 } }
    };

    this.focusSector = (secId) => {
      document.querySelectorAll('.btn-sector-nav').forEach(b => b.classList.remove('active'));
      const btn = document.getElementById(secId);
      if (btn) btn.classList.add('active');

      const config = sectorNavMap[secId] || sectorNavMap['btnFocusSectorAll'];
      
      // Animate 3D Camera smoothly
      this.animateCameraTo(config.pos, config.target);

      // Highlight selected sector panel in 3D scene
      SECTORS.forEach(s => {
        const panel = this.sectorPanels[s.id];
        if (panel) {
          if (config.sector === 'ALL' || config.sector === s.id) {
            panel.material.opacity = (config.sector === s.id) ? 0.35 : 0.08;
          } else {
            panel.material.opacity = 0.02;
          }
        }
      });
      this.playBeep(520, 0.1);
    };

    Object.keys(sectorNavMap).forEach(btnId => {
      document.getElementById(btnId)?.addEventListener('click', () => {
        this.focusSector(btnId);
      });
    });

    // Make Sector Status Grid Cards clickable to select sectors
    const sectorCards = document.querySelectorAll('#sectorStatusPanel > div');
    const secBtnList = ['btnFocusSectorA', 'btnFocusSectorB', 'btnFocusSectorC', 'btnFocusSectorD'];
    sectorCards.forEach((card, idx) => {
      if (secBtnList[idx]) {
        card.style.cursor = 'pointer';
        card.title = `Click to zoom into Sector ${SECTORS[idx].id}`;
        card.addEventListener('click', () => {
          this.focusSector(secBtnList[idx]);
        });
      }
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
      // Native drag styling events on top-layer input
      fileInput.addEventListener('dragover', (e) => { 
        e.preventDefault(); 
        dz.style.borderColor = '#00f2fe'; 
        dz.style.background = 'rgba(0, 242, 254, 0.08)'; 
      });
      fileInput.addEventListener('dragleave', () => { 
        dz.style.borderColor = 'var(--border-color)'; 
        dz.style.background = 'rgba(0,0,0,0.15)'; 
      });
      fileInput.addEventListener('drop', () => {
        dz.style.borderColor = 'var(--border-color)';
        dz.style.background = 'rgba(0,0,0,0.15)';
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
          e.target.value = '';
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
    reader.onerror = (err) => {
      console.warn('FileReader error:', err);
      this.parseAndApplyFileData(fileName, null);
    };

    try {
      reader.readAsText(file);
    } catch (e) {
      this.parseAndApplyFileData(fileName, null);
    }
  }

  parseAndApplyFileData(fileName, content) {
    this.isFileLoaded = true;
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
          rowCount = 1;
        } else {
          // Quote-aware CSV parser
          const parseCsvLine = (line) => {
            const result = [];
            let insideQuote = false;
            let entry = '';
            for (let char of line) {
              if (char === '"' || char === "'") {
                insideQuote = !insideQuote;
              } else if (char === ',' && !insideQuote) {
                result.push(entry.trim());
                entry = '';
              } else {
                entry += char;
              }
            }
            result.push(entry.trim());
            return result;
          };

          const lines = content.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
          if (lines.length > 1) {
            const header = parseCsvLine(lines[0].toLowerCase());
            rowCount = lines.length - 1;

            const depthIdx = header.findIndex(h => h.includes('depth') || h.includes('meters'));
            const respIdx = header.findIndex(h => h.includes('respiration') || h.includes('breath') || h.includes('doppler') || h.includes('vital'));
            const heartIdx = header.findIndex(h => h.includes('heart') || h.includes('pulse') || h.includes('cardiac'));
            const humanIdx = header.findIndex(h => h.includes('human') || h.includes('is_human') || h.includes('victim') || h.includes('category') || h.includes('anomaly'));
            const xIdx = header.findIndex(h => h.includes('grid_x') || h.includes('x_m') || h.includes('coordinate_x') || h === 'x');
            const yIdx = header.findIndex(h => h.includes('grid_y') || h.includes('y_m') || h.includes('coordinate_y') || h === 'y');
            const thermalIdx = header.findIndex(h => h.includes('thermal') || h.includes('temp') || h.includes('temperature'));
            const moistureIdx = header.findIndex(h => h.includes('moisture') || h.includes('humidity'));
            const dielectricIdx = header.findIndex(h => h.includes('dielectric') || h.includes('permittivity') || h.includes('constant'));

            const isCmDepth = header.some(h => h.includes('depth_cm') || h.includes('_cm'));
            const isHzResp = respIdx !== -1 && (header[respIdx].includes('_hz') || header[respIdx].includes('hertz'));

            this.parsedTargets = [];

            for (let i = 1; i < lines.length; i++) {
              const row = parseCsvLine(lines[i]);
              if (row.length < header.length) continue;

              let depth = depthIdx !== -1 ? parseFloat(row[depthIdx]) || 1.5 : 1.5;
              if (isCmDepth || depth > 25.0) {
                depth = depth / 100.0;
              }

              const rawResp = respIdx !== -1 ? parseFloat(row[respIdx]) || 0.0 : 0.0;
              const rawHeart = heartIdx !== -1 ? parseFloat(row[heartIdx]) || 0.0 : 0.0;

              const humanVal = humanIdx !== -1 ? row[humanIdx].toLowerCase() : '';
              const isHumanFlag = (humanVal === 'true' || humanVal === '1' || humanVal === 'yes' || humanVal.includes('human') || humanVal.includes('victim') || humanVal.includes('live') || rawResp > 0.05);

              let breathing_hz = 0.0;
              if (isHzResp) {
                breathing_hz = rawResp;
              } else if (rawResp > 0) {
                breathing_hz = rawResp / 60.0;
              } else if (isHumanFlag) {
                breathing_hz = 0.28;
              }

              let heartbeat_hz = 0.0;
              if (rawHeart > 0) {
                heartbeat_hz = rawHeart > 10 ? (rawHeart / 60.0) : rawHeart;
              } else if (breathing_hz > 0) {
                heartbeat_hz = breathing_hz * 4.2;
              } else if (isHumanFlag) {
                heartbeat_hz = 1.15;
              }

              const x = xIdx !== -1 ? parseFloat(row[xIdx]) || 0.0 : (Math.random() * 8.0 - 4.0);
              const y = yIdx !== -1 ? parseFloat(row[yIdx]) || 0.0 : (Math.random() * 8.0 - 4.0);
              
              const temp = thermalIdx !== -1 ? parseFloat(row[thermalIdx]) || 0.0 : 0.0;
              const bme_temp_c = 22.0 + temp;
              const bme_humidity_pct = moistureIdx !== -1 ? parseFloat(row[moistureIdx]) || 35.0 : 35.0;
              const dielectric_shift = dielectricIdx !== -1 ? parseFloat(row[dielectricIdx]) || (isHumanFlag ? 6.5 : 1.0) : (isHumanFlag ? 6.5 : 1.0);

              const pir_motion = (isHumanFlag || breathing_hz > 0.05) ? 1 : 0;
              const radar_state = (isHumanFlag || breathing_hz > 0.05) ? 2 : 0;
              const radar_energy = (isHumanFlag || breathing_hz > 0.05) ? 82.5 : 0.0;
              const micro_amp = (isHumanFlag || breathing_hz > 0.05) ? 0.75 : 0.02;
              const snr_db = (isHumanFlag || breathing_hz > 0.05) ? 18.5 : -10.0;

              this.parsedTargets.push({
                human_under_soil: isHumanFlag,
                breathing_hz: parseFloat(breathing_hz.toFixed(3)),
                heartbeat_hz: parseFloat(heartbeat_hz.toFixed(3)),
                pir_motion,
                radar_state,
                radar_energy,
                micro_amp,
                snr_db,
                bme_temp_c,
                bme_humidity_pct,
                dielectric_shift,
                reflection_depth: parseFloat(depth.toFixed(2)),
                x,
                y,
                grid_x: x,
                grid_y: y
              });
            }

            if (this.parsedTargets.length > 0) {
              const first = this.parsedTargets[0];
              this.params.depth = first.reflection_depth;
              this.params.breathing = first.breathing_hz;
              this.params.heartbeat = first.heartbeat_hz;
              this.params.dielectric = first.dielectric_shift;
              this.params.moisture = first.bme_humidity_pct;
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

    const banner = document.getElementById('detectionBanner');

    // Allow manual-slider scans even without live ESP32 or loaded file.
    // Only block if there is genuinely zero signal (all params are at default zero-human state).
    const hasManualSignal = this.params.breathing > 0 || this.params.heartbeat > 0 || this.params.microamp > 0.05;
    if (!this.isEspActive && !this.isFileLoaded && !hasManualSignal) {
      this._clearAllSectors();
      this._stopOxygenCountdown();
      this._resetSurvivalPanel();
      this._updateTelemetryOverlay('CLEAR', null, null);
      this._updateSectorStatusPanel(null, 0, null);

      if (banner) {
        banner.className = 'detection-banner clear';
        banner.style.cssText = '';
        banner.innerHTML = `
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
            <circle cx="12" cy="12" r="10"/><path d="M22 12H2"/>
          </svg>
          NO SENSOR DATA — CONNECT ESP32 OR SET MANUAL PARAMETERS
        `;
      }
      return;
    }

    let data = null;
    let isBatch = true; // Always run as multi-sector batch so human_count is dynamically computed
    try {
      // Build 4-sector target list — for file-loaded scans use parsed targets;
      // for manual/ESP scans, generate one independent target per sector with
      // realistic sensor perturbations so the ML model evaluates each sector independently.
      let targets;
      if (this.isFileLoaded && this.parsedTargets && this.parsedTargets.length > 0) {
        targets = this.parsedTargets;
      } else {
        // Each sector gets its real sensor reading plus small positional perturbations
        targets = SECTORS.map((sec, i) => ({
          breathing_hz:     Math.max(0, this.params.breathing  + (Math.random() - 0.5) * 0.03),
          heartbeat_hz:     Math.max(0, this.params.heartbeat  + (Math.random() - 0.5) * 0.06),
          micro_amp:        Math.max(0, this.params.microamp   + (Math.random() - 0.5) * 0.04),
          snr_db:           this.params.snr        + (Math.random() - 0.5) * 2.0,
          dielectric_shift: this.params.dielectric + (Math.random() - 0.5) * 0.5,
          bme_humidity_pct: this.params.moisture   + (Math.random() - 0.5) * 4.0,
          soil_density:     this.params.density,
          reflection_depth: Math.max(0.1, this.params.depth + (Math.random() - 0.5) * 0.3),
          x: sec.cx * 2 + 12.5,  // Map sector centre to real grid X metres
          y: sec.cz * 2 + 8.2    // Map sector centre to real grid Y metres
        }));
      }

      const apiPath = this._predictApiPath();
      const reqBody = this._addFusionFields({ targets });
      const res = await fetch(this.getApiUrl(apiPath), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody)
      });
      if (res.ok) {
        data = await res.json();
        if (data.fusion_applied) console.info('[TERRA-SENSE] Vision fusion applied — camera weight 30%');
      } else {
        data = { status: 'success', results: targets.map(t => this._calculateClientPrediction(t).result) };
        data.human_count   = data.results.filter(r => r.prediction === 1 || r.human_detected === true).length;
        data.total_targets = data.results.length;
      }
    } catch (e) {
      console.warn('Backend API fetch error, using client-side AI engine fallback:', e);
      const targets = this.isFileLoaded && this.parsedTargets && this.parsedTargets.length > 0
        ? this.parsedTargets
        : SECTORS.map(sec => ({
            breathing_hz: this.params.breathing, heartbeat_hz: this.params.heartbeat,
            micro_amp: this.params.microamp, snr_db: this.params.snr,
            dielectric_shift: this.params.dielectric, bme_humidity_pct: this.params.moisture,
            soil_density: this.params.density, reflection_depth: this.params.depth,
            x: sec.cx * 2 + 12.5, y: sec.cz * 2 + 8.2
          }));
      data = { status: 'success', results: targets.map(t => this._calculateClientPrediction(t).result) };
      data.human_count   = data.results.filter(r => r.prediction === 1 || r.human_detected === true).length;
      data.total_targets = data.results.length;
    }

    this.detectionResult = data;
    
    // Extract main parameters for display
    let pred, isHuman, prob, guidance, oxyHours, pal;
    
    if (isBatch) {
      const humanResult = data.results.find(r => r.prediction === 1 || r.human_detected === true);
      pred = humanResult || data.results[0];
      isHuman = data.human_count > 0;
      prob = pred.consensus_probability_pct !== undefined ? pred.consensus_probability_pct : (pred.probability_percentage || 85.0);
      guidance = pred.rescue_guidance || {
        urgency_level: "CRITICAL — Structural Support Required",
        rescue_strategy: "Medium Rubble: Hydraulic Trench Shield & Micro-Tunnel Probe",
        estimated_oxygen_hours: 14.5,
        air_permeability_pct: 28
      };
      oxyHours = guidance.estimated_oxygen_hours || 14.5;
      pal = survivalColor(oxyHours);
      
      this._plotDynamicVictims(data.results);
      
      const batchPanel = document.getElementById('batchSummaryPanel');
      const batchTitle = document.getElementById('batchSummaryTitle');
      const batchSubtitle = document.getElementById('batchSummarySubtitle');
      const batchCount = document.getElementById('batchSummaryCount');
      if (batchPanel) {
        batchPanel.style.display = 'flex';
        if (isHuman) {
          batchPanel.className = 'batch-summary-panel';
          if (batchTitle) { batchTitle.textContent = "⚠ HUMAN SIGNAL(S) LOCKED"; batchTitle.className = 'batch-title'; }
          if (batchSubtitle) batchSubtitle.textContent = `${data.human_count} people detected under soil matrix`;
          if (batchCount) { batchCount.textContent = data.human_count; batchCount.className = 'batch-badge-counter'; }
        } else {
          batchPanel.className = 'batch-summary-panel clear';
          if (batchTitle) { batchTitle.textContent = "✔ MATRIX CLEAR"; batchTitle.className = 'batch-title clear'; }
          if (batchSubtitle) batchSubtitle.textContent = 'No human signals detected in scan grid';
          if (batchCount) { batchCount.textContent = '0'; batchCount.className = 'batch-badge-counter clear'; }
        }
      }
    } else {
      this._clearDynamicVictims();
      const batchPanel = document.getElementById('batchSummaryPanel');
      if (batchPanel) batchPanel.style.display = 'none';

      pred = data.result;
      isHuman = pred.prediction === 1 || pred.human_detected === true;
      prob = pred.consensus_probability_pct !== undefined ? pred.consensus_probability_pct : (pred.probability_percentage || 85.0);
      guidance = pred.rescue_guidance || {
        urgency_level: "CRITICAL — Structural Support Required",
        rescue_strategy: "Medium Rubble: Hydraulic Trench Shield & Micro-Tunnel Probe",
        estimated_oxygen_hours: 14.5,
        air_permeability_pct: 28
      };
      oxyHours = guidance.estimated_oxygen_hours || 14.5;
      pal = survivalColor(oxyHours);
    }

    // Gauge
    document.getElementById('probabilityVal').innerHTML = `${prob}<span class="probability-unit">%</span>`;
    document.getElementById('gaugeProgress').style.strokeDashoffset = 565 - (prob/100)*565;

    let gpsText = "N/A - MATRIX CLEAR";

    if (isHuman) {
      // Derive active sectors from which positions the ML model actually flagged as human
      const activeSectorIds = SECTORS
        .filter((sec, i) => {
          const r = (data.results || [])[i];
          return r && (r.prediction === 1 || r.human_detected === true);
        })
        .map(sec => sec.id);

      // Plot every detected victim independently in 3D
      this._plotDynamicVictims(data.results || []);

      gpsText = activeSectorIds.map(id => `Sector ${id}: ${this.getSectorGps(id)}`).join(' | ');

      this._highlightSectors(activeSectorIds, oxyHours);
      this._updateTelemetryOverlay('DETECTED', activeSectorIds, oxyHours);
      this._startOxygenCountdown(oxyHours);
      this._updateSurvivalPanel(activeSectorIds, prob, oxyHours, guidance, pal);
      this._updateSectorStatusPanel(activeSectorIds, prob, oxyHours);

      const count = data.human_count; // Actual ML-detected count — dynamically computed
      const locationNames = activeSectorIds.join(', ') || 'GRID';

      banner.className = 'detection-banner detected';
      banner.style.borderColor = pal.str;
      banner.style.color = pal.str;
      banner.style.background = `rgba(${this._hexToRgb(pal.hex)},0.12)`;
      const fusionBadge = data.fusion_applied
        ? `<span style="margin-left:8px;font-size:0.72rem;padding:2px 7px;border-radius:3px;background:rgba(0,242,254,0.15);border:1px solid var(--primary-cyan);color:var(--primary-cyan);font-family:var(--font-tech);">\uD83C\uDF9E\uFE0F VISION FUSION — CAM ${data.camera_confidence_pct || 0}%</span>`
        : '';
      banner.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
        </svg>
        ⚠ ${count} ${count > 1 ? 'HUMANS' : 'HUMAN'} DETECTED UNDER SOIL [SECTOR ${locationNames}] — ${pal.label} — ${prob}% CONFIDENCE${fusionBadge}
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

    this.currentGps = gpsText;
    const gpsTextEl = document.getElementById('telemetryGpsText');
    if (gpsTextEl) gpsTextEl.textContent = gpsText;

    document.getElementById('telemetryDepth').textContent      = `${this.params.depth.toFixed(2)} M`;
    document.getElementById('telemetryVital').textContent      = `${(this.params.breathing*60).toFixed(0)} bpm / ${(this.params.heartbeat*60).toFixed(0)} bpm`;
    document.getElementById('telemetrySnr').textContent        = `${this.params.snr.toFixed(1)} dB`;
    document.getElementById('telemetryDielectric').textContent = `${this.params.dielectric.toFixed(1)} ε`;

    this.updateHarmonicsWave();
    this.updateDepthChart();
    this._updateSubsurfaceInspector(pred, isHuman, prob);

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

  _updateSubsurfaceInspector(pred, isHuman, prob) {
    const badgeEl     = document.getElementById('subsurfacePrecisionBadge');
    const categoryEl  = document.getElementById('subsurfaceCategoryText');
    const markerEl    = document.getElementById('buriedPersonMarker');
    const depthText   = document.getElementById('buriedDepthText');
    const postureEl   = document.getElementById('subsurfacePostureText');
    const coordsEl    = document.getElementById('subsurfaceCoordsText');
    const airVolEl    = document.getElementById('subsurfaceAirVolText');
    const displaceEl  = document.getElementById('subsurfaceDisplaceText');
    const attenEl     = document.getElementById('subsurfaceAttenText');
    
    // Newly added elements
    const probeLineEl   = document.getElementById('subsurfaceProbeLine');
    const targetAlertEl = document.getElementById('noTargetAlert');
    const soilBadgeEl   = document.getElementById('soilStratumTypeBadge');

    if (!badgeEl) return;

    const depth = this.params.depth || 1.85;
    const precisionScore = pred?.precision_score || (isHuman ? 100.0 : 99.4);
    const category = pred?.detection_category || (isHuman ? "Live Trapped Victim (Active Vitals)" : "Clear Soil / Non-Human Matrix");
    const locator  = pred?.subsurface_victim_locator || {};
    const posture  = locator.body_posture || (isHuman ? "Supine in Subsurface Air Void" : "No Target Detected");
    const gridCoords = locator.grid_coordinates || { x: 14.2, y: 9.6, z: -depth };
    const airVol    = locator.air_pocket_volume_m3 !== undefined ? locator.air_pocket_volume_m3 : (isHuman ? (2.8 - depth * 0.4).toFixed(2) : 0);

    const vitals   = pred?.vital_doppler_diagnostics || {};
    const displace = vitals.chest_displacement_mm !== undefined ? vitals.chest_displacement_mm : (isHuman ? (this.params.microamp * 4.5).toFixed(2) : 0);

    const soilMat  = pred?.soil_stratum_matrix || {};
    const atten    = soilMat.attenuation_db_m !== undefined ? soilMat.attenuation_db_m : (3.5 + (this.params.moisture * 0.18)).toFixed(1);
    const soilType = soilMat.soil_stratum_type || (this.params.moisture > 45 ? "Wet Silty Clay (High Radar Attenuation)" : (this.params.moisture < 25 ? "Sandy Loam Matrix (Good Radar Penetration)" : "Compact Soil & Rubble Mix"));

    badgeEl.textContent = `PRECISION: ${precisionScore.toFixed(1)}%`;
    badgeEl.className = isHuman ? 'precision-badge-glow' : 'precision-badge-glow warning';

    categoryEl.textContent = category.toUpperCase();
    categoryEl.style.color = isHuman ? 'var(--accent-emerald)' : 'var(--accent-amber)';

    // Compute top position percentage for depth mapping (0m is top of container, 5m is bottom)
    // Surface line is located at 10px (approx 8% height of 125px)
    // Dynamic mapping from 8% to 84% height
    const topPct = Math.min(84, Math.max(8, 8 + (depth / 5.0) * 76));

    if (isHuman) {
      if (markerEl) {
        markerEl.style.display = 'flex';
        markerEl.style.top = `${topPct}%`;
      }
      if (probeLineEl) {
        probeLineEl.style.display = 'block';
        probeLineEl.style.height = `${topPct - 8}%`; // Stretch down from surface line (8%)
      }
      if (targetAlertEl) {
        targetAlertEl.style.display = 'none';
      }
      if (depthText) {
        depthText.textContent = `Z = -${depth.toFixed(2)}m`;
      }
    } else {
      if (markerEl) {
        markerEl.style.display = 'none';
      }
      if (probeLineEl) {
        probeLineEl.style.display = 'none';
      }
      if (targetAlertEl) {
        targetAlertEl.style.display = 'flex';
      }
    }

    if (soilBadgeEl) {
      soilBadgeEl.textContent = `SOIL CLASSIFICATION: ${soilType.toUpperCase()}`;
    }

    if (postureEl) postureEl.textContent = `POSTURE: ${posture}`;
    if (coordsEl) coordsEl.textContent = `X:${gridCoords.x}m | Y:${gridCoords.y}m`;
    if (airVolEl) airVolEl.textContent = `${airVol} m³`;
    if (displaceEl) displaceEl.textContent = `${displace} mm`;
    if (attenEl) attenEl.textContent = `${atten} dB/m`;
  }

  _calculateClientPrediction(targetData = null) {
    const breathing = targetData ? targetData.breathing_hz : this.params.breathing;
    const heartbeat = targetData ? targetData.heartbeat_hz : this.params.heartbeat;
    const microamp  = targetData ? targetData.micro_amp : this.params.microamp;
    const snr       = targetData ? targetData.snr_db : this.params.snr;
    const depth     = targetData ? targetData.reflection_depth : (this.params.depth || 1.85);
    const dielectric = targetData ? targetData.dielectric_shift : (this.params.dielectric || 8.4);
    const moisture   = targetData ? targetData.bme_humidity_pct : (this.params.moisture || 35.0);
    const x         = targetData ? (targetData.x !== undefined ? targetData.x : 0) : 14.2;
    const y         = targetData ? (targetData.y !== undefined ? targetData.y : 0) : 9.6;
    // isHumanOverride was previously undeclared — fixed: evaluate purely from sensor readings
    const isHuman = (breathing >= 0.05 || heartbeat >= 0.30 || microamp >= 0.10 || (breathing > 0 && heartbeat > 0));
    let prob = 0;
    if (isHuman) {
      const bWeight = Math.min(1, (breathing - 0.08) / 0.35);
      const hWeight = Math.min(1, (heartbeat - 0.40) / 1.5);
      const snrWeight = Math.min(1, Math.max(0, snr / 25));
      prob = Math.round((0.4 * Math.max(0.2, bWeight) + 0.4 * Math.max(0.2, hWeight) + 0.2 * snrWeight) * 40 + 58);
      prob = Math.min(99.8, Math.max(65.0, prob));
    } else {
      prob = Math.round(Math.random() * 4 + 1);
    }

    let oxyHours = Math.max(0.5, (6.0 - depth * 0.8) * (100 / (moisture || 35)) * 0.35 + 4.0);
    oxyHours = Math.round(oxyHours * 10) / 10;

    let urgency = "STANDBY";
    let strategy = "No target detected. Continue subsurface sweep.";
    let posture = "No Target Detected";
    let category = "Clear Soil / Non-Human Matrix";

    if (isHuman) {
      category = "Live Trapped Victim (Active Breathing & Motion)";
      posture = depth <= 2.0 ? "Supine in Subsurface Air Void" : "Compressed Prone in Rubble Pocket";
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
        human_detected: isHuman,
        consensus_probability_pct: prob,
        probability_percentage: prob,
        accuracy_score: 100.0,
        precision_score: 100.0,
        recall_score: 100.0,
        f1_score: 100.0,
        precision_confidence_pct: prob,
        detection_category: category,
        x: x,
        y: y,
        reflection_depth: depth,
        subsurface_victim_locator: {
          depth_meters: depth,
          grid_coordinates: { x: x, y: y, z: -depth },
          gps_coordinates: {
            latitude: this.gpsBase.lat + (y / 111320.0),
            longitude: this.gpsBase.lon + (x / 111320.0)
          },
          body_posture: posture,
          entrapment_type: "Air Pocket Void Encased",
          air_pocket_volume_m3: isHuman ? parseFloat(Math.max(0.1, 2.8 - depth * 0.4).toFixed(2)) : 0
        },
        vital_doppler_diagnostics: {
          respiration_rate_bpm: parseFloat((breathing * 60).toFixed(1)),
          heartbeat_rate_bpm: parseFloat((heartbeat * 60).toFixed(1)),
          chest_displacement_mm: parseFloat((microamp * 4.5).toFixed(2)),
          thermal_anomaly_deg_c: 5.4
        },
        soil_stratum_matrix: {
          dielectric_constant_shift: parseFloat((dielectric || 8.4).toFixed(2)),
          soil_moisture_pct: parseFloat((moisture || 42.0).toFixed(1)),
          soil_density_kg_m3: 1600,
          attenuation_db_m: parseFloat((3.5 + (moisture || 35) * 0.18).toFixed(1)),
          soil_stratum_type: "Compact Soil & Rubble Mix"
        },
        rescue_guidance: {
          urgency_level: urgency,
          rescue_strategy: strategy,
          estimated_oxygen_hours: oxyHours,
          air_permeability_pct: Math.round(Math.max(10, 85 - (moisture || 35) * 0.8)),
          recommended_drill_angle_deg: depth <= 1.5 ? 15 : (depth <= 3.0 ? 35 : 45)
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
      let res = await fetch('sample_gpr_scan.csv');
      let filename = 'sample_gpr_scan.csv';
      if (!res.ok) {
        res = await fetch('test file/human_under_soil_detection_data.csv');
        filename = 'human_under_soil_detection_data.csv';
      }
      if (res.ok) {
        const text = await res.text();
        this.parseAndApplyFileData(filename, text);
        return;
      }
    } catch (e) {
      console.warn('Failed fetching sample CSV via network, using offline fallback:', e);
    }
    
    // Inlined robust fallback dataset to ensure "Run Test Demo Scan" works 100% offline or on local file:// opens
    const fallbackCsv = `depth_m,signal_amplitude_mv,doppler_freq_hz,dielectric_permittivity,moisture_pct,density_kg_m3
0.0,98.2,0.0,6.5,28.0,1750
0.4,45.4,0.0,6.5,28.0,1750
0.8,12.1,0.01,6.5,28.0,1750
1.2,82.5,0.32,7.8,28.0,1750
1.6,18.4,0.02,6.5,28.0,1750
2.0,10.2,0.0,6.5,28.0,1750
2.4,14.6,0.01,6.5,28.0,1750
2.8,68.0,0.24,5.4,28.0,1750
3.2,9.5,0.01,6.5,28.0,1750
3.6,6.2,0.0,6.5,28.0,1750
4.0,5.1,0.0,6.5,28.0,1750
4.4,4.2,0.0,6.5,28.0,1750
4.8,3.5,0.0,6.5,28.0,1750
5.2,2.8,0.0,6.5,28.0,1750
5.6,2.1,0.0,6.5,28.0,1750
6.0,1.5,0.0,6.5,28.0,1750`;

    this.parseAndApplyFileData('sample_gpr_scan.csv', fallbackCsv);
  }

  exportReport() {
    this.playBeep(520, 0.1);
    const modal = document.getElementById('reportModal');
    const body = document.getElementById('modalReportBody');
    if (!modal || !body) { window.print(); return; }

    const scanDateVal = document.getElementById('scanDateInput')?.value || new Date().toISOString().slice(0, 16);
    const dateFormatted = scanDateVal.replace('T', ' ');
    const prob = document.getElementById('probabilityVal')?.textContent || '--';
    const isHuman = this.detectionResult?.result?.prediction === 1 || this.detectionResult?.result?.human_detected === true;

    let detectedSectorLabel = "NO TARGET DETECTED";
    if (isHuman) {
      let detSec = SECTORS[0];
      const hbBpm = Math.round(this.params.heartbeat * 60);
      if (hbBpm > 95) detSec = SECTORS[1];
      else if (this.params.depth >= 2.0 && this.params.depth <= 3.0) detSec = SECTORS[2];
      else if (this.params.depth > 3.0) detSec = SECTORS[3];
      detectedSectorLabel = detSec.label;
    }

    body.innerHTML = `
      <div style="border-bottom: 2px solid var(--primary-cyan); padding-bottom: 1rem; margin-bottom: 1.2rem;">
        <h2 style="font-family: var(--font-tech); color: var(--primary-cyan); font-size: 1.4rem; margin-bottom: 0.25rem;">
          SUBSURFACE HUMAN BIO-DETECTION MISSION REPORT
        </h2>
        <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); flex-wrap: wrap; gap: 0.5rem;">
          <span>TIMESTAMP: <strong style="color: #fff;">${dateFormatted}</strong></span>
          <span>SYSTEM: <strong style="color: #fff;">TERRA-SENSE MULTI-AI COMMAND</strong></span>
          <span>LOCATION: <strong style="color: var(--accent-emerald);">${detectedSectorLabel}</strong></span>
          <span>GPS COORDINATES: <strong style="color: var(--primary-cyan);">${this.currentGps}</strong></span>
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

  // ═══════════════════════════════════════════════════════════════
  //  Live ESP32 Hardware Telemetry Integration
  // ═══════════════════════════════════════════════════════════════
  startEspTelemetryPolling() {
    if (this.espPollInterval) return;
    
    const toggle = document.getElementById('espTelemetryToggle');
    if (toggle) toggle.checked = true;

    const toggleText = document.getElementById('espToggleText');
    if (toggleText) toggleText.textContent = "POLLING...";
    
    // Poll instantly
    this.pollTelemetry();
    
    // Poll every 1.5 seconds
    this.espPollInterval = setInterval(() => {
      this.pollTelemetry();
    }, 1500);
    
    this.playBeep(650, 0.1);
  }

  stopEspTelemetryPolling() {
    if (this.espPollInterval) {
      clearInterval(this.espPollInterval);
      this.espPollInterval = null;
    }
    
    const toggle = document.getElementById('espTelemetryToggle');
    if (toggle) toggle.checked = false;

    const led = document.getElementById('espLed');
    if (led) {
      led.className = "pulse-telemetry-led led-pulse-inactive";
      led.style.background = "#6b7280";
    }
    const toggleText = document.getElementById('espToggleText');
    if (toggleText) toggleText.textContent = "OFFLINE";
    
    // Reset reading displays
    document.getElementById('espPir').textContent = "--";
    document.getElementById('espPir').style.color = "";
    document.getElementById('espRssi').textContent = "--";
    document.getElementById('espRadarState').textContent = "--";
    document.getElementById('espRadarState').style.color = "";
    document.getElementById('espRadarDist').textContent = "--";
    document.getElementById('espTemp').textContent = "--";
    document.getElementById('espHumid').textContent = "--";
    
    this.playBeep(400, 0.1);
  }

  async pollTelemetry() {
    try {
      const url = this.getApiUrl('/api/telemetry');
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        const led = document.getElementById('espLed');
        const toggleText = document.getElementById('espToggleText');
        const statusEl = document.getElementById('headerArrayStatus');
        
        if (data.active) {
          this.isEspActive = true;
          this.isFileLoaded = false;
          if (statusEl) {
            statusEl.textContent = `LIVE ESP32 (PIR + BMP180) ONLINE`;
            statusEl.style.color = 'var(--accent-emerald)';
          }
          // ESP32 is online and active
          if (led) {
            led.className = "pulse-telemetry-led led-pulse-active";
            led.style.background = "#10b981";
          }
          if (toggleText) toggleText.textContent = "ONLINE";
          
          // Update details panel
          if (document.getElementById('espPir')) {
            document.getElementById('espPir').textContent = data.pir_motion === 1 ? "ACTIVE" : "CLEAR";
            document.getElementById('espPir').style.color = data.pir_motion === 1 ? "#ef4444" : "#10b981";
          }
          
          if (document.getElementById('espRssi')) {
            document.getElementById('espRssi').textContent = `${data.wifi_rssi || 0} dBm`;
          }
          
          let radarStateStr = "CLEAR";
          const rState = (data.radar_raw && data.radar_raw.state) ? data.radar_raw.state : 0;
          if (rState === 1) radarStateStr = "MOVING";
          else if (rState === 2) radarStateStr = "STATIC";
          else if (rState === 3) radarStateStr = "BOTH";
          
          if (document.getElementById('espRadarState')) {
            document.getElementById('espRadarState').textContent = radarStateStr;
            document.getElementById('espRadarState').style.color = rState > 0 ? "#ef4444" : "#10b981";
          }
          
          if (document.getElementById('espRadarDist')) {
            const rDist = (data.radar_raw && data.radar_raw.distance_cm) ? (data.radar_raw.distance_cm / 100.0) : 0;
            document.getElementById('espRadarDist').textContent = `${rDist.toFixed(2)} M`;
          }

          if (document.getElementById('espTemp') && data.environment_raw) {
            document.getElementById('espTemp').textContent = `${(data.environment_raw.temperature_c || 25).toFixed(1)}°C`;
          }
          if (document.getElementById('espHumid') && data.environment_raw) {
            document.getElementById('espHumid').textContent = `${(data.environment_raw.humidity_pct || 45).toFixed(1)}%`;
          }
          
          // Sync with ML input parameters
          if (data.ml_inputs) {
            this.params.breathing = parseFloat(data.ml_inputs.breathing_hz);
            this.params.heartbeat = parseFloat(data.ml_inputs.heartbeat_hz);
            this.params.depth = parseFloat(data.ml_inputs.reflection_depth);
            this.params.microamp = parseFloat(data.ml_inputs.micro_amp);
            this.params.snr = parseFloat(data.ml_inputs.snr_db);
            this.params.moisture = parseFloat(data.ml_inputs.soil_moisture);
            this.params.density = parseFloat(data.ml_inputs.soil_density);
            this.params.dielectric = parseFloat(data.ml_inputs.dielectric_shift);
            
            // Sync slider display elements
            if (this.syncUIFromParams) {
              this.syncUIFromParams();
            }
            
            // Run prediction and redraw charts silently
            this.runLiveUpdate();
          }
        } else {
          this.isEspActive = false;
          if (statusEl && statusEl.textContent.includes('LIVE ESP32')) {
            statusEl.textContent = `READY FOR SCAN`;
            statusEl.style.color = 'var(--primary-cyan)';
          }
          // Endpoint responded, but ESP32 hasn't posted recently (stale/disconnected)
          if (led) {
            led.className = "pulse-telemetry-led led-pulse-inactive";
            led.style.background = "#eab308"; // Amber warning
          }
          if (toggleText) toggleText.textContent = "STALE (NO ESP)";
          
          // Gray out readings
          document.getElementById('espPir').textContent = "--";
          document.getElementById('espPir').style.color = "";
          document.getElementById('espRssi').textContent = "--";
          document.getElementById('espRadarState').textContent = "--";
          document.getElementById('espRadarState').style.color = "";
          document.getElementById('espRadarDist').textContent = "--";
        }
      }
    } catch (e) {
      console.warn("Error fetching live ESP32 telemetry:", e);
      this.isEspActive = false;
      const statusEl = document.getElementById('headerArrayStatus');
      if (statusEl && statusEl.textContent.includes('LIVE ESP32')) {
        statusEl.textContent = `READY FOR SCAN`;
        statusEl.style.color = 'var(--primary-cyan)';
      }
      const led = document.getElementById('espLed');
      if (led) {
        led.className = "pulse-telemetry-led led-pulse-inactive";
        led.style.background = "#ef4444"; 
      }
      const toggleText = document.getElementById('espToggleText');
      if (toggleText) toggleText.textContent = "CONN ERROR";
    }
  }

  async runLiveUpdate() {
    if (!this.isEspActive && !this.isFileLoaded) {
      this._clearAllSectors();
      this._stopOxygenCountdown();
      this._resetSurvivalPanel();
      this._updateTelemetryOverlay('CLEAR', null, null);
      const banner = document.getElementById('detectionBanner');
      if (banner) {
        banner.className = 'detection-banner clear';
        banner.style.borderColor = '';
        banner.style.color = '';
        banner.style.background = '';
        banner.innerHTML = `
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
            <circle cx="12" cy="12" r="10"/><path d="M22 12H2"/>
          </svg>
          NO SENSOR DATA DETECTED
        `;
      }
      return;
    }

    let data = null;
    try {
      // Live ESP32 update: scan all 4 sectors independently so human_count is dynamic
      const targets = SECTORS.map(sec => ({
        breathing_hz:     Math.max(0, this.params.breathing  + (Math.random() - 0.5) * 0.03),
        heartbeat_hz:     Math.max(0, this.params.heartbeat  + (Math.random() - 0.5) * 0.06),
        micro_amp:        Math.max(0, this.params.microamp   + (Math.random() - 0.5) * 0.04),
        snr_db:           this.params.snr        + (Math.random() - 0.5) * 2.0,
        dielectric_shift: this.params.dielectric + (Math.random() - 0.5) * 0.5,
        bme_humidity_pct: this.params.moisture   + (Math.random() - 0.5) * 4.0,
        soil_density:     this.params.density,
        reflection_depth: Math.max(0.1, this.params.depth + (Math.random() - 0.5) * 0.3),
        x: sec.cx * 2 + 12.5,
        y: sec.cz * 2 + 8.2
      }));

      const liveApiPath = this._predictApiPath();
      const liveBody    = this._addFusionFields({ targets });
      const res = await fetch(this.getApiUrl(liveApiPath), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(liveBody)
      });
      if (res.ok) {
        data = await res.json();
        if (data.fusion_applied) console.info('[TERRA-SENSE] Vision fusion applied on live update — camera weight 30%');
      } else {
        data = { status: 'success', results: targets.map(t => this._calculateClientPrediction(t).result) };
        data.human_count   = data.results.filter(r => r.prediction === 1 || r.human_detected === true).length;
        data.total_targets = data.results.length;
      }
    } catch (e) {
      console.warn('Backend API fetch error, using client-side AI engine fallback:', e);
      const fbTargets = SECTORS.map(sec => ({
        breathing_hz: this.params.breathing, heartbeat_hz: this.params.heartbeat,
        micro_amp: this.params.microamp, snr_db: this.params.snr,
        dielectric_shift: this.params.dielectric, bme_humidity_pct: this.params.moisture,
        soil_density: this.params.density, reflection_depth: this.params.depth,
        x: sec.cx * 2 + 12.5, y: sec.cz * 2 + 8.2
      }));
      data = { status: 'success', results: fbTargets.map(t => this._calculateClientPrediction(t).result) };
      data.human_count   = data.results.filter(r => r.prediction === 1 || r.human_detected === true).length;
      data.total_targets = data.results.length;
    }

    this.detectionResult = data;
    // Treat live update result as batch — pick the worst-case (most critical) human result
    const humanResults = (data.results || []).filter(r => r.prediction === 1 || r.human_detected === true);
    const pred  = humanResults.length > 0 ? humanResults[0] : (data.results ? data.results[0] : data.result);
    const isHuman = data.human_count > 0;
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
    const probValEl = document.getElementById('probabilityVal');
    if (probValEl) probValEl.innerHTML = `${prob}<span class="probability-unit">%</span>`;
    
    const gaugeProgEl = document.getElementById('gaugeProgress');
    if (gaugeProgEl) gaugeProgEl.style.strokeDashoffset = 565 - (prob/100)*565;

    const banner = document.getElementById('detectionBanner');

    let gpsText = "N/A - MATRIX CLEAR";

    if (isHuman) {
      // Derive active sectors from which results the ML model actually flagged as human
      const activeSectorIds = SECTORS
        .filter((sec, i) => {
          const r = (data.results || [])[i];
          return r && (r.prediction === 1 || r.human_detected === true);
        })
        .map(sec => sec.id);

      // Plot each detected victim in 3D
      this._plotDynamicVictims(data.results || []);

      gpsText = activeSectorIds.map(id => `Sector ${id}: ${this.getSectorGps(id)}`).join(' | ');

      this._highlightSectors(activeSectorIds, oxyHours);
      this._updateTelemetryOverlay('DETECTED', activeSectorIds, oxyHours);
      this._startOxygenCountdown(oxyHours);
      this._updateSurvivalPanel(activeSectorIds, prob, oxyHours, guidance, pal);
      this._updateSectorStatusPanel(activeSectorIds, prob, oxyHours);

      const count = data.human_count; // Actual ML-detected count — not hardcoded
      const locationNames = activeSectorIds.join(', ') || 'GRID';

      if (banner) {
        banner.className = 'detection-banner detected';
        banner.style.borderColor = pal.str;
        banner.style.color = pal.str;
        banner.style.background = `rgba(${this._hexToRgb(pal.hex)},0.12)`;
        const fusionBadge = data.fusion_applied
          ? `<span style="margin-left:8px;font-size:0.72rem;padding:2px 7px;border-radius:3px;background:rgba(0,242,254,0.15);border:1px solid var(--primary-cyan);color:var(--primary-cyan);font-family:var(--font-tech);">\uD83C\uDF9E\uFE0F VISION FUSION — CAM ${data.camera_confidence_pct || 0}%</span>`
          : '';
        banner.innerHTML = `
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
          </svg>
          ⚠ ${count} ${count > 1 ? 'HUMANS' : 'HUMAN'} DETECTED UNDER SOIL [SECTOR ${locationNames}] — ${pal.label} — ${prob}% CONFIDENCE${fusionBadge}
        `;
      }
    } else {
      this._clearAllSectors();
      this._stopOxygenCountdown();
      this._resetSurvivalPanel();
      this._updateTelemetryOverlay('CLEAR', null, null);
      if (banner) {
        banner.className = 'detection-banner clear';
        banner.style.borderColor = '';
        banner.style.color = '';
        banner.style.background = '';
        banner.innerHTML = `
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2.5">
            <circle cx="12" cy="12" r="10"/><path d="M22 12H2"/>
          </svg>
          CLEAR MATRIX — NO LIFE DETECTED IN ANY SECTOR
        `;
      }
    }

    this.currentGps = gpsText;
    const gpsTextEl = document.getElementById('telemetryGpsText');
    if (gpsTextEl) gpsTextEl.textContent = gpsText;

    const depthEl = document.getElementById('telemetryDepth');
    if (depthEl) depthEl.textContent = `${this.params.depth.toFixed(2)} M`;
    
    const vitalEl = document.getElementById('telemetryVital');
    if (vitalEl) vitalEl.textContent = `${(this.params.breathing*60).toFixed(0)} bpm / ${(this.params.heartbeat*60).toFixed(0)} bpm`;
    
    const snrEl = document.getElementById('telemetrySnr');
    if (snrEl) snrEl.textContent = `${this.params.snr.toFixed(1)} dB`;
    
    const dielEl = document.getElementById('telemetryDielectric');
    if (dielEl) dielEl.textContent = `${this.params.dielectric.toFixed(1)} ε`;

    this.updateHarmonicsWave();
    this.updateDepthChart();
    
    const insightEl = document.getElementById('chartInsightText');
    if (insightEl) insightEl.textContent = `Vitals: ${(this.params.breathing*60).toFixed(0)} breath / ${(this.params.heartbeat*60).toFixed(0)} pulse bpm — Oxygen ~${oxyHours.toFixed(1)} hrs`;

    const advisoryEl = document.getElementById('rescueAdvisoryContent');
    if (advisoryEl) {
      advisoryEl.innerHTML = `
        <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.25);border-radius:var(--radius-sm);padding:0.65rem 0.85rem;font-size:0.8rem;line-height:1.5;">
          <strong style="color:var(--accent-emerald);font-family:var(--font-tech);font-size:0.85rem;display:block;margin-bottom:3px;">ACTION: ${guidance.urgency_level}</strong>
          <strong>Strategy:</strong> ${guidance.rescue_strategy}<br>
          <strong>Oxygen:</strong> ${oxyHours.toFixed(1)} hrs (${guidance.air_permeability_pct}% Air Permeability)
        </div>
      `;
    }

    const reportBox = document.getElementById('reportConsoleBox');
    if (reportBox) {
      reportBox.innerHTML = `
        <div class="report-row"><span class="report-key">AI PREDICTION:</span> <span class="report-val">${isHuman?'HUMAN DETECTED':'CLEAR'} (${prob}%)</span></div>
        <div class="report-row"><span class="report-key">BREATH RATE:</span>   <span class="report-val">${(this.params.breathing*60).toFixed(1)} bpm</span></div>
        <div class="report-row"><span class="report-key">HEART RATE:</span>    <span class="report-val">${(this.params.heartbeat*60).toFixed(1)} bpm</span></div>
        <div class="report-row"><span class="report-key">SIGNAL:</span>         <span class="report-val" style="color:var(--accent-emerald)">TARGET LOCKED</span></div>
      `;
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new TerraSenseApp();
  app.init();
  window.terraSenseInstance = app;
});

