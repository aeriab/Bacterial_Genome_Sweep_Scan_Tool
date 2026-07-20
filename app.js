// Genome Sweep Scan Browser
// Static, client-side rendering of per-species CNN sweep-scan predictions.
// Data files (webapp/data/<species>.bin + <species>_manifest.json + species_index.json)
// are produced by export_scan_data.py from plot_inference/genome_scans/.

(() => {
  'use strict';

  const DATA_DIR = 'data';
  const LABEL_NEUTRAL = 0, LABEL_HARD = 1, LABEL_SOFT = 2;
  const GENE_ANNOTATIONS_KEY = 'genomeScanBrowser.geneAnnotations.v1';

  // ---------------------------------------------------------------------
  // Persistent UI state (survives species switches; only viewport resets).
  // ---------------------------------------------------------------------
  const state = {
    binSize: 30,
    annotateHard: false,
    hardThreshold: 20,
    annotateSoft: false,
    softThreshold: 20,
    species: null,
    viewport: null, // {xMin, xMax} in continuous genome coordinate for the current species
  };

  const speciesCache = new Map();   // species -> {manifest, position(Float64Array), pNeutral, pHard, pSoft, label(Uint8Array)}
  const pooledCache = new Map();    // "species|binSize" -> {y(Float32Array), colorCode(Uint8Array), yMax}
  const runsCache = new Map();      // "species|binSize|label|threshold" -> [{startIdx,endIdx,xStart,xEnd,length}]

  let speciesIndex = [];            // [{species, n_windows, n_contigs, x_max}]

  // species -> [{id, contigNum, startBp, endBp, text, color}]
  const geneAnnotations = loadGeneAnnotations();

  // ---------------------------------------------------------------------
  // DOM refs
  // ---------------------------------------------------------------------
  const canvas = document.getElementById('plot-canvas');
  let ctx = canvas.getContext('2d');
  const loadingEl = document.getElementById('loading');
  const tooltipEl = document.getElementById('tooltip');
  const speciesInfoEl = document.getElementById('species-info');

  const searchInput = document.getElementById('species-search');
  const dropdownEl = document.getElementById('species-dropdown');

  const poolSlider = document.getElementById('pool-slider');
  const poolNumber = document.getElementById('pool-number');
  const poolValueLabel = document.getElementById('pool-value');
  const poolPluralLabel = document.getElementById('pool-plural');

  const annotateHardCb = document.getElementById('annotate-hard');
  const hardThresholdInput = document.getElementById('hard-threshold');
  const annotateSoftCb = document.getElementById('annotate-soft');
  const softThresholdInput = document.getElementById('soft-threshold');

  const resetViewBtn = document.getElementById('reset-view-btn');
  const exportImageBtn = document.getElementById('export-image-btn');

  const legendHardRunEl = document.getElementById('legend-hard-run');
  const legendSoftRunEl = document.getElementById('legend-soft-run');
  const topbarLegendEl = document.getElementById('topbar-legend');

  const geneContigSelect = document.getElementById('gene-contig');
  const geneStartInput = document.getElementById('gene-start');
  const geneEndInput = document.getElementById('gene-end');
  const geneTextInput = document.getElementById('gene-text');
  const geneColorInput = document.getElementById('gene-color');
  const geneAddBtn = document.getElementById('gene-add-btn');
  const geneErrorEl = document.getElementById('gene-error');
  const geneListEl = document.getElementById('gene-list');

  // ---------------------------------------------------------------------
  // Formatting helpers
  // ---------------------------------------------------------------------
  function prettySpeciesName(key) {
    // "Bacteroides_ovatus_58035" -> "Bacteroides ovatus (58035)"
    const m = key.match(/^(.*)_(\d+)$/);
    if (!m) return key.replace(/_/g, ' ');
    return `${m[1].replace(/_/g, ' ')} (${m[2]})`;
  }

  function formatBp(bp) {
    const abs = Math.abs(bp);
    if (abs >= 1e6) return (bp / 1e6).toFixed(2) + ' Mb';
    if (abs >= 1e3) return (bp / 1e3).toFixed(1) + ' kb';
    return Math.round(bp).toLocaleString() + ' bp';
  }

  function formatBpExact(bp) {
    return Math.round(bp).toLocaleString() + ' bp';
  }

  function labelName(code) {
    return code === LABEL_HARD ? 'Hard sweep' : code === LABEL_SOFT ? 'Soft sweep' : 'Neutral';
  }

  // "Nice" tick step, d3-style.
  function niceTickStep(range, targetCount) {
    if (range <= 0) return 1;
    const rough = range / Math.max(targetCount, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const norm = rough / mag;
    let step;
    if (norm < 1.5) step = 1;
    else if (norm < 3.5) step = 2;
    else if (norm < 7.5) step = 5;
    else step = 10;
    return step * mag;
  }

  // ---------------------------------------------------------------------
  // Gene / region annotations (user-entered, persisted to localStorage)
  // ---------------------------------------------------------------------
  function loadGeneAnnotations() {
    try {
      const raw = localStorage.getItem(GENE_ANNOTATIONS_KEY);
      if (!raw) return new Map();
      const obj = JSON.parse(raw);
      return new Map(Object.entries(obj));
    } catch (e) {
      console.warn('Could not load saved gene annotations:', e);
      return new Map();
    }
  }

  function saveGeneAnnotations() {
    try {
      const obj = Object.fromEntries(geneAnnotations.entries());
      localStorage.setItem(GENE_ANNOTATIONS_KEY, JSON.stringify(obj));
    } catch (e) {
      console.warn('Could not save gene annotations:', e);
    }
  }

  function getGeneAnnotationsFor(species) {
    return geneAnnotations.get(species) || [];
  }

  function addGeneAnnotation(species, annotation) {
    const list = geneAnnotations.get(species) || [];
    list.push(annotation);
    geneAnnotations.set(species, list);
    saveGeneAnnotations();
  }

  function removeGeneAnnotation(species, id) {
    const list = geneAnnotations.get(species) || [];
    const next = list.filter(a => a.id !== id);
    geneAnnotations.set(species, next);
    saveGeneAnnotations();
  }

  function populateGeneContigSelect(entry) {
    geneContigSelect.innerHTML = '';
    for (const c of entry.manifest.contigs) {
      const opt = document.createElement('option');
      opt.value = c.contig_num;
      opt.textContent = `contig ${c.contig_num} (${formatBp(c.x_end - c.x_start)})`;
      geneContigSelect.appendChild(opt);
    }
  }

  function renderGeneList() {
    const list = state.species ? getGeneAnnotationsFor(state.species) : [];
    geneListEl.innerHTML = '';
    for (const a of list) {
      const row = document.createElement('div');
      row.className = 'gene-list-item';
      const sw = document.createElement('span');
      sw.className = 'sw';
      sw.style.background = a.color;
      const label = document.createElement('span');
      label.className = 'gene-label';
      label.textContent = `${a.text} — contig ${a.contigNum}: ${formatBpExact(a.startBp)}–${formatBpExact(a.endBp)}`;
      label.title = label.textContent;
      const delBtn = document.createElement('button');
      delBtn.textContent = '×';
      delBtn.title = 'Remove annotation';
      delBtn.addEventListener('click', () => {
        removeGeneAnnotation(state.species, a.id);
        renderGeneList();
        scheduleRender();
      });
      row.appendChild(sw);
      row.appendChild(label);
      row.appendChild(delBtn);
      geneListEl.appendChild(row);
    }
  }

  function showGeneError(msg) {
    geneErrorEl.textContent = msg;
    geneErrorEl.classList.remove('hidden');
  }
  function hideGeneError() {
    geneErrorEl.classList.add('hidden');
  }

  geneAddBtn.addEventListener('click', () => {
    hideGeneError();
    if (!state.species) { showGeneError('Load a species first.'); return; }
    const contigNum = geneContigSelect.value;
    const startBp = Math.round(+geneStartInput.value);
    const endBp = Math.round(+geneEndInput.value);
    const text = geneTextInput.value.trim();
    const color = geneColorInput.value;

    if (!contigNum) { showGeneError('Select a contig.'); return; }
    if (!Number.isFinite(startBp) || !Number.isFinite(endBp)) { showGeneError('Enter numeric start/end bp.'); return; }
    if (startBp < 0 || startBp >= endBp) { showGeneError('Start bp must be >= 0 and less than end bp.'); return; }
    if (!text) { showGeneError('Enter a label.'); return; }

    const entry = speciesCache.get(state.species);
    const contig = entry.manifest.contigs.find(c => c.contig_num === contigNum);
    const contigLen = contig ? contig.x_end - contig.x_start : Infinity;
    if (contig && endBp > contigLen) {
      showGeneError(`End bp exceeds contig ${contigNum}'s length (~${Math.round(contigLen).toLocaleString()} bp).`);
      return;
    }

    addGeneAnnotation(state.species, {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      contigNum, startBp, endBp, text, color,
    });
    geneTextInput.value = '';
    renderGeneList();
    scheduleRender();
  });

  // ---------------------------------------------------------------------
  // CSS variable colors (read once per render so theme changes propagate to canvas)
  // ---------------------------------------------------------------------
  function readColors() {
    const s = getComputedStyle(document.documentElement);
    return {
      surface1: s.getPropertyValue('--surface-1').trim(),
      surface2: s.getPropertyValue('--surface-2').trim(),
      textPrimary: s.getPropertyValue('--text-primary').trim(),
      textSecondary: s.getPropertyValue('--text-secondary').trim(),
      textMuted: s.getPropertyValue('--text-muted').trim(),
      gridline: s.getPropertyValue('--gridline').trim(),
      baseline: s.getPropertyValue('--baseline').trim(),
      neutral: s.getPropertyValue('--series-neutral').trim(),
      hard: s.getPropertyValue('--series-hard').trim(),
      soft: s.getPropertyValue('--series-soft').trim(),
      bandHard: s.getPropertyValue('--band-hard').trim(),
      bandHardEdge: s.getPropertyValue('--band-hard-edge').trim(),
      bandSoft: s.getPropertyValue('--band-soft').trim(),
      bandSoftEdge: s.getPropertyValue('--band-soft-edge').trim(),
      contigA: s.getPropertyValue('--contig-bg-a').trim(),
      contigB: s.getPropertyValue('--contig-bg-b').trim(),
    };
  }

  // ---------------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------------
  async function loadSpeciesIndex() {
    const resp = await fetch(`${DATA_DIR}/species_index.json`);
    speciesIndex = await resp.json();
  }

  async function loadSpeciesData(species) {
    if (speciesCache.has(species)) return speciesCache.get(species);

    const [manifest, buf] = await Promise.all([
      fetch(`${DATA_DIR}/${species}_manifest.json`).then(r => r.json()),
      fetch(`${DATA_DIR}/${species}.bin`).then(r => r.arrayBuffer()),
    ]);

    const n = manifest.n_windows;
    const positionF32 = new Float32Array(buf, 0, n);
    // Promote to Float64 for precise cross-contig coordinate math.
    const position = Float64Array.from(positionF32);
    const pNeutral = new Float32Array(buf, 4 * n, n);
    const pHard = new Float32Array(buf, 8 * n, n);
    const pSoft = new Float32Array(buf, 12 * n, n);
    const label = new Uint8Array(buf, 16 * n, n);

    const entry = { manifest, position, pNeutral, pHard, pSoft, label };
    speciesCache.set(species, entry);
    return entry;
  }

  // ---------------------------------------------------------------------
  // Rolling mean (matches pandas .rolling(window=w, center=True, min_periods=1).mean())
  // computed independently within each contig.
  // ---------------------------------------------------------------------
  function rollingMeanInto(out, values, startIdx, endIdx, w) {
    const len = endIdx - startIdx;
    if (len <= 0) return;
    // Matches pandas .rolling(window=w, center=True).mean(): for even w the
    // extra element sits before the center index, not after.
    const before = Math.floor(w / 2);
    const after = w - 1 - before;
    const prefix = new Float64Array(len + 1);
    for (let i = 0; i < len; i++) prefix[i + 1] = prefix[i] + values[startIdx + i];
    for (let i = 0; i < len; i++) {
      let lo = i - before; if (lo < 0) lo = 0;
      let hi = i + after; if (hi > len - 1) hi = len - 1;
      const cnt = hi - lo + 1;
      out[startIdx + i] = (prefix[hi + 1] - prefix[lo]) / cnt;
    }
  }

  function getPooled(species) {
    const key = `${species}|${state.binSize}`;
    if (pooledCache.has(key)) return pooledCache.get(key);

    const entry = speciesCache.get(species);
    const n = entry.position.length;
    const pnPooled = new Float32Array(n);
    const phPooled = new Float32Array(n);
    const psPooled = new Float32Array(n);

    for (const c of entry.manifest.contigs) {
      rollingMeanInto(pnPooled, entry.pNeutral, c.start_index, c.end_index, state.binSize);
      rollingMeanInto(phPooled, entry.pHard, c.start_index, c.end_index, state.binSize);
      rollingMeanInto(psPooled, entry.pSoft, c.start_index, c.end_index, state.binSize);
    }

    const y = new Float32Array(n);
    const colorCode = new Uint8Array(n);
    let yMax = 0.5;
    for (let i = 0; i < n; i++) {
      const vn = pnPooled[i], vh = phPooled[i], vs = psPooled[i];
      let cc;
      if (vn > 0.01) {
        cc = LABEL_NEUTRAL;
      } else if (vn >= vs && vn >= vh) {
        cc = LABEL_NEUTRAL;
      } else if (vs >= vh) {
        cc = LABEL_SOFT;
      } else {
        cc = LABEL_HARD;
      }
      // Source predictions_NNN.txt files record P_Neutral to only 6 decimal
      // places, so any true value below 5e-7 gets truncated to the literal
      // string "0.000000". Flooring at 1e-10 (as if that were real measured
      // precision) invented a fake y=10 spike with an empty gap between it
      // and the true 6-decimal resolution limit (y=6). 5e-7 is the honest
      // floor: the midpoint of the last representable rounding bucket.
      const yv = -Math.log10(Math.max(vn, 5e-7));
      y[i] = yv;
      colorCode[i] = cc;
      if (yv > yMax) yMax = yv;
    }

    const result = { y, colorCode, yMax: Math.min(10, yMax * 1.08) };
    pooledCache.set(key, result);
    return result;
  }

  // Runs are defined over the same pooled colorCode that determines each
  // dot's on-screen color (getPooled), not the raw per-window CNN label —
  // otherwise the highlighted bands wouldn't correspond to what's actually
  // plotted (raw single-window calls are noisy and don't line up with the
  // smoothed/pooled signal a user reads as "a run of hard/soft dots").
  function getRuns(species, targetLabel, threshold) {
    const key = `${species}|${state.binSize}|${targetLabel}|${threshold}`;
    if (runsCache.has(key)) return runsCache.get(key);

    const entry = speciesCache.get(species);
    const pooled = getPooled(species);
    const runs = [];
    for (const c of entry.manifest.contigs) {
      let runStart = -1;
      for (let i = c.start_index; i < c.end_index; i++) {
        if (pooled.colorCode[i] === targetLabel) {
          if (runStart === -1) runStart = i;
        } else {
          if (runStart !== -1) {
            const runLen = i - runStart;
            if (runLen >= threshold) {
              runs.push({ startIdx: runStart, endIdx: i - 1, xStart: entry.position[runStart], xEnd: entry.position[i - 1], length: runLen });
            }
            runStart = -1;
          }
        }
      }
      if (runStart !== -1) {
        const runLen = c.end_index - runStart;
        if (runLen >= threshold) {
          runs.push({ startIdx: runStart, endIdx: c.end_index - 1, xStart: entry.position[runStart], xEnd: entry.position[c.end_index - 1], length: runLen });
        }
      }
    }
    runsCache.set(key, runs);
    return runs;
  }

  // Binary search: first index with position >= target
  function lowerBound(arr, target) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------
  const MARGIN = { top: 28, right: 20, bottom: 36, left: 60 };
  let renderScheduled = false;

  function scheduleRender() {
    if (renderScheduled) return;
    renderScheduled = true;
    requestAnimationFrame(() => {
      renderScheduled = false;
      render();
    });
  }

  function resizeCanvasToDisplaySize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, Math.floor(rect.width));
    const h = Math.max(150, Math.floor(rect.height));
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
    }
    return { w, h, dpr };
  }

  function render() {
    if (!state.species || !speciesCache.has(state.species)) return;
    const { w, h, dpr } = resizeCanvasToDisplaySize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderCore(w, h);
  }

  // Draws the full chart into whatever `ctx` currently points at, using a
  // (w, h) viewport in CSS-pixel units. Shared by the live on-screen render
  // and the high-resolution PNG export (which temporarily swaps `ctx` to an
  // offscreen canvas scaled up by EXPORT_SCALE).
  function renderCore(w, h) {
    ctx.clearRect(0, 0, w, h);

    const colors = readColors();
    ctx.fillStyle = colors.surface2;
    ctx.fillRect(0, 0, w, h);

    const entry = speciesCache.get(state.species);
    const pooled = getPooled(state.species);
    const { xMin, xMax } = state.viewport;
    const plotW = w - MARGIN.left - MARGIN.right;
    const plotH = h - MARGIN.top - MARGIN.bottom;
    if (plotW <= 0 || plotH <= 0) return;

    const yMax = pooled.yMax;
    const xToPx = (x) => MARGIN.left + ((x - xMin) / (xMax - xMin)) * plotW;
    const yToPx = (y) => MARGIN.top + plotH - (Math.min(y, yMax) / yMax) * plotH;

    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
    ctx.clip();

    // --- contig background bands ---
    const contigs = entry.manifest.contigs;
    contigs.forEach((c, i) => {
      if (c.x_end < xMin || c.x_start > xMax) return;
      ctx.fillStyle = i % 2 === 0 ? colors.contigA : colors.contigB;
      const x0 = xToPx(Math.max(c.x_start, xMin));
      const x1 = xToPx(Math.min(c.x_end, xMax));
      ctx.fillRect(x0, MARGIN.top, Math.max(1, x1 - x0), plotH);
    });

    // --- annotation bands ---
    if (state.annotateHard) {
      const runs = getRuns(state.species, LABEL_HARD, state.hardThreshold);
      drawRuns(runs, colors.bandHard, colors.bandHardEdge, xMin, xMax, xToPx, plotH);
    }
    if (state.annotateSoft) {
      const runs = getRuns(state.species, LABEL_SOFT, state.softThreshold);
      drawRuns(runs, colors.bandSoft, colors.bandSoftEdge, xMin, xMax, xToPx, plotH);
    }

    // --- user gene/region annotations ---
    const genes = getGeneAnnotationsFor(state.species)
      .map(a => geneAnnotationPixels(a, entry, xToPx, xMin, xMax))
      .filter(Boolean);
    for (const g of genes) {
      ctx.fillStyle = hexToRgba(g.color, 0.16);
      ctx.strokeStyle = hexToRgba(g.color, 0.7);
      ctx.lineWidth = 1.5;
      ctx.fillRect(g.x0, MARGIN.top, Math.max(1.5, g.x1 - g.x0), plotH);
      ctx.beginPath();
      ctx.moveTo(g.x0, MARGIN.top); ctx.lineTo(g.x0, MARGIN.top + plotH);
      ctx.moveTo(g.x1, MARGIN.top); ctx.lineTo(g.x1, MARGIN.top + plotH);
      ctx.stroke();
    }

    // --- gridlines (y) ---
    ctx.strokeStyle = colors.gridline;
    ctx.lineWidth = 1;
    const yTickStep = niceTickStep(yMax, 5);
    ctx.beginPath();
    for (let yv = 0; yv <= yMax + 1e-9; yv += yTickStep) {
      const py = Math.round(yToPx(yv)) + 0.5;
      ctx.moveTo(MARGIN.left, py);
      ctx.lineTo(MARGIN.left + plotW, py);
    }
    ctx.stroke();

    // --- scatter points, batched by color ---
    const lo = lowerBound(entry.position, xMin - (xMax - xMin) * 0.02);
    const hi = lowerBound(entry.position, xMax + (xMax - xMin) * 0.02);
    const buckets = [[], [], []]; // neutral, hard, soft
    for (let i = lo; i < hi; i++) buckets[pooled.colorCode[i]].push(i);

    const colorFor = [colors.neutral, colors.hard, colors.soft];
    const radiusFor = [1.1, 1.45, 1.45];
    for (let cc = 0; cc < 3; cc++) {
      if (!buckets[cc].length) continue;
      ctx.fillStyle = colorFor[cc];
      const r = radiusFor[cc];
      ctx.beginPath();
      for (const i of buckets[cc]) {
        const px = xToPx(entry.position[i]);
        const py = yToPx(pooled.y[i]);
        ctx.moveTo(px + r, py);
        ctx.arc(px, py, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }

    // gene labels drawn last (within the clip) so they sit above the dots
    ctx.font = '600 11.5px system-ui, sans-serif';
    for (const g of genes) {
      drawGeneLabel(g, MARGIN, plotW, plotH, colors);
    }

    ctx.restore();

    // --- axes ---
    drawYAxis(colors, yMax, yTickStep, yToPx, plotH);
    drawXAxis(colors, entry, xMin, xMax, xToPx, w, h);

    // --- frame ---
    ctx.strokeStyle = colors.baseline;
    ctx.lineWidth = 1;
    ctx.strokeRect(MARGIN.left + 0.5, MARGIN.top + 0.5, plotW - 1, plotH - 1);

    updateInfoPanels(entry);
  }

  function drawRuns(runs, fill, edge, xMin, xMax, xToPx, plotH) {
    ctx.fillStyle = fill;
    ctx.strokeStyle = edge;
    ctx.lineWidth = 1.5;
    for (const run of runs) {
      if (run.xEnd < xMin || run.xStart > xMax) continue;
      const x0 = xToPx(Math.max(run.xStart, xMin));
      const x1 = xToPx(Math.min(run.xEnd, xMax));
      const width = Math.max(1.5, x1 - x0);
      ctx.fillRect(x0, MARGIN.top, width, plotH);
      ctx.beginPath();
      ctx.moveTo(x0, MARGIN.top);
      ctx.lineTo(x0, MARGIN.top + plotH);
      ctx.moveTo(x1, MARGIN.top);
      ctx.lineTo(x1, MARGIN.top + plotH);
      ctx.stroke();
    }
  }

  function hexToRgba(hex, alpha) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!m) return hex;
    const r = parseInt(m[1], 16), g = parseInt(m[2], 16), b = parseInt(m[3], 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function geneAnnotationPixels(a, entry, xToPx, xMin, xMax) {
    const contig = entry.manifest.contigs.find(c => c.contig_num === a.contigNum);
    if (!contig) return null;
    const xStartData = contig.x_start + a.startBp;
    const xEndData = contig.x_start + a.endBp;
    if (xEndData < xMin || xStartData > xMax) return null;
    return {
      x0: xToPx(Math.max(xStartData, xMin)),
      x1: xToPx(Math.min(xEndData, xMax)),
      color: a.color,
      text: a.text,
    };
  }

  function drawGeneLabel(g, MARGIN, plotW, plotH, colors) {
    const textW = ctx.measureText(g.text).width;
    const padX = 5, padY = 3;
    const boxH = 15;
    const y = MARGIN.top + 14; // fixed near-top position within the plot
    let boxX0 = g.x1 + 6;
    let textAlignLeft = true;
    if (boxX0 + textW + padX * 2 > MARGIN.left + plotW) {
      // not enough room to the right; try the left side of the band instead
      boxX0 = g.x0 - 6 - (textW + padX * 2);
      textAlignLeft = false;
      if (boxX0 < MARGIN.left) { boxX0 = Math.max(MARGIN.left, g.x1 + 6); textAlignLeft = true; }
    }
    ctx.fillStyle = hexToRgba(colors.surface1, 0.88);
    ctx.fillRect(boxX0, y - boxH / 2, textW + padX * 2, boxH);
    ctx.fillStyle = g.color;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(g.text, boxX0 + padX, y + 1);
  }

  function drawYAxis(colors, yMax, yTickStep, yToPx, plotH) {
    ctx.fillStyle = colors.textSecondary;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const decimals = yTickStep < 0.1 ? 2 : yTickStep < 1 ? 1 : 0;
    for (let yv = 0; yv <= yMax + 1e-9; yv += yTickStep) {
      const py = yToPx(yv);
      ctx.fillText(yv.toFixed(decimals), MARGIN.left - 8, py);
    }
    ctx.save();
    ctx.translate(14, MARGIN.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = colors.textMuted;
    drawSubscriptRun(ctx, [
      { text: '-log', sub: false },
      { text: '10', sub: true },
      { text: '(P', sub: false },
      { text: 'Neutral', sub: true },
      { text: ')', sub: false },
    ], 0, 0, '11px system-ui, sans-serif', '8px system-ui, sans-serif', 3);
    ctx.restore();
  }

  // Draws a run of text fragments centered at (cx, baselineY), where fragments
  // flagged `sub` render in a smaller font with their baseline shifted down
  // (true typographic subscripts, e.g. the "10" in log10 or "Neutral" in
  // P_Neutral) rather than full-size text or a unicode-subscript approximation.
  function drawSubscriptRun(targetCtx, segments, cx, baselineY, baseFont, subFont, subDy) {
    const widths = segments.map(seg => {
      targetCtx.font = seg.sub ? subFont : baseFont;
      return targetCtx.measureText(seg.text).width;
    });
    const total = widths.reduce((a, b) => a + b, 0);
    let x = cx - total / 2;
    const prevAlign = targetCtx.textAlign;
    const prevBaseline = targetCtx.textBaseline;
    targetCtx.textAlign = 'left';
    targetCtx.textBaseline = 'alphabetic';
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      targetCtx.font = seg.sub ? subFont : baseFont;
      targetCtx.fillText(seg.text, x, seg.sub ? baselineY + subDy : baselineY);
      x += widths[i];
    }
    targetCtx.textAlign = prevAlign;
    targetCtx.textBaseline = prevBaseline;
  }

  function drawXAxis(colors, entry, xMin, xMax, xToPx, w, h) {
    const plotBottom = h - MARGIN.bottom;
    const contigs = entry.manifest.contigs;
    const visible = contigs.filter(c => c.x_end >= xMin && c.x_start <= xMax);

    ctx.font = '10.5px system-ui, sans-serif';
    ctx.fillStyle = colors.textMuted;

    // contig boundary labels, drawn in the reserved header strip above the
    // plot (y in [0, MARGIN.top]) so they never overlap scatter points.
    ctx.strokeStyle = colors.gridline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top - 0.5);
    ctx.lineTo(w - MARGIN.right, MARGIN.top - 0.5);
    ctx.stroke();
    visible.forEach((c) => {
      const segX0 = xToPx(Math.max(c.x_start, xMin));
      const segX1 = xToPx(Math.min(c.x_end, xMax));
      const segW = segX1 - segX0;
      if (segW > 34) {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = colors.textSecondary;
        const label = segW > 90 ? `contig ${c.contig_num}` : c.contig_num;
        ctx.fillText(label, (segX0 + segX1) / 2, MARGIN.top / 2 + 2);
      }
      if (segX0 > MARGIN.left + 1) {
        ctx.strokeStyle = colors.baseline;
        ctx.beginPath();
        ctx.moveTo(segX0, 2);
        ctx.lineTo(segX0, MARGIN.top - 2);
        ctx.stroke();
      }
    });

    // per-contig bp ticks, only for contigs wide enough on screen
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = colors.textMuted;
    ctx.beginPath();
    visible.forEach((c) => {
      const segX0 = xToPx(Math.max(c.x_start, xMin));
      const segX1 = xToPx(Math.min(c.x_end, xMax));
      const segW = segX1 - segX0;
      if (segW < 70) return;

      const bpLo = Math.max(0, xMin - c.x_start);
      const bpHi = Math.min(c.x_end - c.x_start, xMax - c.x_start);
      const bpRange = bpHi - bpLo;
      if (bpRange <= 0) return;
      const targetTicks = Math.max(2, Math.floor(segW / 90));
      const step = niceTickStep(bpRange, targetTicks);
      const firstTick = Math.ceil(bpLo / step) * step;
      for (let bp = firstTick; bp <= bpHi; bp += step) {
        const cx = c.x_start + bp;
        if (cx < xMin || cx > xMax) continue;
        const px = xToPx(cx);
        ctx.moveTo(px, plotBottom);
        ctx.lineTo(px, plotBottom + 5);
        ctx.fillText(formatBp(bp), px, plotBottom + 8);
      }
    });
    ctx.strokeStyle = colors.baseline;
    ctx.stroke();
  }

  function updateInfoPanels(entry) {
    const m = entry.manifest;
    speciesInfoEl.textContent = `${prettySpeciesName(m.species)} — ${m.n_windows.toLocaleString()} windows, ${m.contigs.length} contig${m.contigs.length === 1 ? '' : 's'}`;
  }

  // ---------------------------------------------------------------------
  // Species selection
  // ---------------------------------------------------------------------
  async function selectSpecies(species) {
    loadingEl.classList.remove('hidden');
    try {
      const entry = await loadSpeciesData(species);
      state.species = species;
      state.viewport = { xMin: 0, xMax: entry.manifest.x_max };
      searchInput.value = prettySpeciesName(species);
      dropdownEl.classList.add('hidden');
      populateGeneContigSelect(entry);
      renderGeneList();
      scheduleRender();
    } finally {
      loadingEl.classList.add('hidden');
    }
  }

  function renderDropdown(filterText) {
    const q = filterText.trim().toLowerCase();
    const matches = (q
      ? speciesIndex.filter(s => prettySpeciesName(s.species).toLowerCase().includes(q) || s.species.toLowerCase().includes(q))
      : speciesIndex
    ).slice(0, 40);

    dropdownEl.innerHTML = '';
    if (!matches.length) {
      dropdownEl.classList.add('hidden');
      return;
    }
    for (const s of matches) {
      const row = document.createElement('div');
      row.className = 'species-option';
      const nameSpan = document.createElement('span');
      nameSpan.textContent = prettySpeciesName(s.species);
      const metaSpan = document.createElement('span');
      metaSpan.className = 'meta';
      metaSpan.textContent = `${s.n_windows.toLocaleString()} win · ${s.n_contigs} contig${s.n_contigs === 1 ? '' : 's'}`;
      row.appendChild(nameSpan);
      row.appendChild(metaSpan);
      row.addEventListener('mousedown', (e) => {
        e.preventDefault();
        selectSpecies(s.species);
      });
      dropdownEl.appendChild(row);
    }
    dropdownEl.classList.remove('hidden');
  }

  searchInput.addEventListener('focus', () => renderDropdown(searchInput.value === (state.species ? prettySpeciesName(state.species) : '') ? '' : searchInput.value));
  searchInput.addEventListener('input', () => renderDropdown(searchInput.value));
  searchInput.addEventListener('blur', () => setTimeout(() => dropdownEl.classList.add('hidden'), 120));
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.species-picker')) dropdownEl.classList.add('hidden');
  });

  // ---------------------------------------------------------------------
  // Controls
  // ---------------------------------------------------------------------
  function setBinSize(v) {
    v = Math.max(1, Math.min(300, Math.round(v)));
    state.binSize = v;
    poolSlider.value = v;
    poolNumber.value = v;
    poolValueLabel.textContent = v;
    poolPluralLabel.textContent = v === 1 ? '' : 's';
    scheduleRender();
  }
  poolSlider.addEventListener('input', () => setBinSize(+poolSlider.value));
  poolNumber.addEventListener('change', () => setBinSize(+poolNumber.value));

  function setLegendRunVisibility() {
    legendHardRunEl.classList.toggle('hidden', !state.annotateHard);
    legendSoftRunEl.classList.toggle('hidden', !state.annotateSoft);
    // Only-soft-on is the one case with an awkward gap (the reserved-but-
    // hidden hard-run slot would sit between "Soft sweep" and "Soft-run
    // region"), so swap the pair's order then; any time hard-run is on,
    // switch back to the normal Hard-run-then-Soft-run order.
    topbarLegendEl.classList.toggle('swap-run-order', state.annotateSoft && !state.annotateHard);
  }

  annotateHardCb.addEventListener('change', () => { state.annotateHard = annotateHardCb.checked; setLegendRunVisibility(); scheduleRender(); });
  annotateSoftCb.addEventListener('change', () => { state.annotateSoft = annotateSoftCb.checked; setLegendRunVisibility(); scheduleRender(); });
  hardThresholdInput.addEventListener('change', () => {
    state.hardThreshold = Math.max(2, Math.round(+hardThresholdInput.value) || 2);
    hardThresholdInput.value = state.hardThreshold;
    scheduleRender();
  });
  softThresholdInput.addEventListener('change', () => {
    state.softThreshold = Math.max(2, Math.round(+softThresholdInput.value) || 2);
    softThresholdInput.value = state.softThreshold;
    scheduleRender();
  });

  resetViewBtn.addEventListener('click', () => {
    if (!state.species) return;
    const entry = speciesCache.get(state.species);
    state.viewport = { xMin: 0, xMax: entry.manifest.x_max };
    scheduleRender();
  });

  // ---------------------------------------------------------------------
  // High-resolution image export (for figures) — reuses renderCore() by
  // temporarily pointing the module-level `ctx` at an offscreen canvas
  // supersampled by EXPORT_SCALE, so exported PNGs stay crisp when scaled
  // up or printed rather than just capturing the on-screen pixel size.
  // ---------------------------------------------------------------------
  const EXPORT_SCALE = 4;

  function exportImage() {
    if (!state.species || !speciesCache.has(state.species)) return;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(200, Math.round(rect.width));
    const h = Math.max(150, Math.round(rect.height));

    const off = document.createElement('canvas');
    off.width = Math.round(w * EXPORT_SCALE);
    off.height = Math.round(h * EXPORT_SCALE);
    const offCtx = off.getContext('2d');
    offCtx.setTransform(EXPORT_SCALE, 0, 0, EXPORT_SCALE, 0, 0);

    const liveCtx = ctx;
    ctx = offCtx;
    try {
      renderCore(w, h);
    } finally {
      ctx = liveCtx;
    }

    off.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      a.href = url;
      a.download = `${state.species}_sweep_scan_${stamp}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  exportImageBtn.addEventListener('click', exportImage);

  // Initialize control DOM from state defaults.
  setBinSize(state.binSize);
  annotateHardCb.checked = state.annotateHard;
  hardThresholdInput.value = state.hardThreshold;
  annotateSoftCb.checked = state.annotateSoft;
  softThresholdInput.value = state.softThreshold;
  setLegendRunVisibility();

  // ---------------------------------------------------------------------
  // Zoom / pan
  // ---------------------------------------------------------------------
  const MIN_SPAN_BP = 200;

  function zoomAt(dataX, factor) {
    const { xMin, xMax } = state.viewport;
    const entry = speciesCache.get(state.species);
    const fullSpan = entry.manifest.x_max;
    let newMin = dataX - (dataX - xMin) * factor;
    let newMax = dataX + (xMax - dataX) * factor;
    let span = newMax - newMin;
    if (span < MIN_SPAN_BP) {
      const mid = (newMin + newMax) / 2;
      newMin = mid - MIN_SPAN_BP / 2;
      newMax = mid + MIN_SPAN_BP / 2;
      span = MIN_SPAN_BP;
    }
    if (span > fullSpan) { newMin = 0; newMax = fullSpan; }
    else {
      if (newMin < 0) { newMax -= newMin; newMin = 0; }
      if (newMax > fullSpan) { newMin -= (newMax - fullSpan); newMax = fullSpan; }
      newMin = Math.max(0, newMin);
    }
    state.viewport = { xMin: newMin, xMax: newMax };
  }

  function panBy(dataDx) {
    const entry = speciesCache.get(state.species);
    const fullSpan = entry.manifest.x_max;
    let { xMin, xMax } = state.viewport;
    const span = xMax - xMin;
    let newMin = xMin + dataDx;
    let newMax = xMax + dataDx;
    if (newMin < 0) { newMax -= newMin; newMin = 0; }
    if (newMax > fullSpan) { newMin -= (newMax - fullSpan); newMax = fullSpan; }
    newMin = Math.max(0, newMin);
    state.viewport = { xMin: newMin, xMax: newMax };
  }

  function pixelToDataScale() {
    const rect = canvas.getBoundingClientRect();
    const plotW = rect.width - MARGIN.left - MARGIN.right;
    const { xMin, xMax } = state.viewport;
    return (xMax - xMin) / plotW;
  }

  function canvasDataXFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left - MARGIN.left;
    const { xMin, xMax } = state.viewport;
    const plotW = rect.width - MARGIN.left - MARGIN.right;
    return xMin + (px / plotW) * (xMax - xMin);
  }

  canvas.addEventListener('wheel', (e) => {
    if (!state.species) return;
    e.preventDefault();
    const scale = pixelToDataScale();
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY) && !e.shiftKey) {
      panBy(e.deltaX * scale);
    } else if (e.shiftKey) {
      panBy(e.deltaY * scale);
    } else {
      const dataX = canvasDataXFromEvent(e);
      const factor = Math.pow(1.0016, e.deltaY);
      zoomAt(dataX, factor);
    }
    scheduleRender();
  }, { passive: false });

  let dragging = false;
  let dragLastX = 0;
  canvas.addEventListener('pointerdown', (e) => {
    if (!state.species) return;
    dragging = true;
    dragLastX = e.clientX;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (dragging) {
      const dx = e.clientX - dragLastX;
      dragLastX = e.clientX;
      const scale = pixelToDataScale();
      panBy(-dx * scale);
      scheduleRender();
      hideTooltip();
    } else {
      showTooltip(e);
    }
  });
  canvas.addEventListener('pointerup', (e) => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener('pointerleave', () => { dragging = false; hideTooltip(); });

  function hideTooltip() { tooltipEl.classList.add('hidden'); }

  function nearestWindowIndex(entry, dataX) {
    const idx = lowerBound(entry.position, dataX);
    let best = idx;
    if (idx > 0 && (idx >= entry.position.length || Math.abs(entry.position[idx - 1] - dataX) < Math.abs(entry.position[idx] - dataX))) {
      best = idx - 1;
    }
    if (best < 0 || best >= entry.position.length) return -1;
    return best;
  }

  function showTooltip(e) {
    if (!state.species || dragging) return;
    const entry = speciesCache.get(state.species);
    const pooled = getPooled(state.species);
    const dataX = canvasDataXFromEvent(e);
    const { xMin, xMax } = state.viewport;
    if (dataX < xMin || dataX > xMax) { hideTooltip(); return; }

    const best = nearestWindowIndex(entry, dataX);
    if (best < 0) { hideTooltip(); return; }

    const contig = entry.manifest.contigs.find(c => best >= c.start_index && best < c.end_index);
    if (!contig) { hideTooltip(); return; }
    const bpLocal = entry.position[best] - contig.x_start;

    const rect = canvas.parentElement.getBoundingClientRect();
    tooltipEl.innerHTML = '';
    const lines = [
      `contig ${contig.contig_num} @ ${formatBpExact(bpLocal)}`,
      `raw call: ${labelName(entry.label[best])}`,
      `P_Neutral=${entry.pNeutral[best].toFixed(3)}  P_Hard=${entry.pHard[best].toFixed(3)}  P_Soft=${entry.pSoft[best].toFixed(3)}`,
      `pooled (${state.binSize}w): -log10(P_N)=${pooled.y[best].toFixed(2)}, class=${labelName(pooled.colorCode[best])}`,
    ];
    for (const line of lines) {
      const div = document.createElement('div');
      div.textContent = line;
      tooltipEl.appendChild(div);
    }
    let left = e.clientX - rect.left + 14;
    let top = e.clientY - rect.top + 14;
    tooltipEl.classList.remove('hidden');
    const tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
    if (left + tw > rect.width) left = e.clientX - rect.left - tw - 14;
    if (top + th > rect.height) top = e.clientY - rect.top - th - 14;
    tooltipEl.style.left = left + 'px';
    tooltipEl.style.top = top + 'px';
  }

  window.addEventListener('resize', scheduleRender);
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', scheduleRender);
  }

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  (async function init() {
    await loadSpeciesIndex();
    if (speciesIndex.length) {
      await selectSpecies(speciesIndex[0].species);
    }
  })();
})();
