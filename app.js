// Genome Sweep Scan Browser
// Static, client-side rendering of per-species CNN sweep-scan predictions.
// Data files (webapp/data/<species>.bin + <species>_manifest.json + species_index.json)
// are produced by export_scan_data.py from plot_inference/genome_scans/.

(() => {
  'use strict';

  const DATA_DIR = 'data';
  const LABEL_NEUTRAL = 0, LABEL_HARD = 1, LABEL_SOFT = 2;
  const GENE_ANNOTATIONS_KEY = 'genomeScanBrowser.geneAnnotations.v1';
  // Band colors for curated H12 peaks, by sweep type — deliberately neon and
  // well clear of the CNN hard/soft red/blue. Also written into data/peaks.json
  // by build_peaks_json.py; forced here too so a stale JSON can't override them.
  const PEAK_COLOR_HARD = '#ff6a00';   // neon orange
  const PEAK_COLOR_SOFT = '#b026ff';   // neon purple
  const peakColor = (kind) => (kind === 'soft' ? PEAK_COLOR_SOFT : PEAK_COLOR_HARD);

  // ---------------------------------------------------------------------
  // Persistent UI state (survives species switches; only viewport resets).
  // ---------------------------------------------------------------------
  const state = {
    binSize: 30,
    annotateSweeps: false,
    sweepThreshold: 20,
    showPeaks: false,
    species: null,
    viewport: null, // {xMin, xMax} in continuous genome coordinate for the current species
    // Drag positions for the two PNG-export overlays, each {fx, fy} as a
    // fraction of the canvas (top-left corner of the box). null = leave the
    // box in its default top-right stacked spot.
    overlayPos: { caption: null, legend: null },
    hapSites: null,      // parsed data/<species>_haplotype_sites.json, or null
    hapSelected: null,   // key of the currently shown haplotype snapshot
  };

  const speciesCache = new Map();   // species -> {manifest, position(Float64Array), pNeutral, pHard, pSoft, label(Uint8Array)}
  const pooledCache = new Map();    // "species|binSize" -> {y(Float32Array), colorCode(Uint8Array), yMax}
  const runsCache = new Map();      // "species|binSize|threshold" -> [{startIdx,endIdx,xStart,xEnd,length,kind}]

  let speciesIndex = [];            // [{species, n_windows, n_contigs, x_max}]

  // Parsed data/peaks.json: { species -> [{peak, contigNum, startBp, endBp, kind, text, color}] }.
  // Curated H12 sweep peaks, built by build_peaks_json.py. Loaded once at boot;
  // {} if the file is missing.
  let peaksData = {};

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

  const annotateSweepsCb = document.getElementById('annotate-sweeps');
  const sweepThresholdInput = document.getElementById('sweep-threshold');
  const showPeaksCb = document.getElementById('show-peaks');
  const peakCountNoteEl = document.getElementById('peak-count-note');
  const legendPeakHardEl = document.getElementById('legend-peak-hard');
  const legendPeakSoftEl = document.getElementById('legend-peak-soft');

  const resetViewBtn = document.getElementById('reset-view-btn');
  const exportImageBtn = document.getElementById('export-image-btn');
  const exportCaptionToggle = document.getElementById('export-caption-toggle');
  const exportLegendToggle = document.getElementById('export-legend-toggle');

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

  const canvasWrapEl = document.querySelector('.plot-canvas-wrap');
  const hapPanelEl = document.getElementById('hap-panel');
  const hapPanelNoteEl = document.getElementById('hap-panel-note');
  const hapToggleBtn = document.getElementById('hap-toggle');
  const hapChipsEl = document.getElementById('hap-chips');
  const hapFigureEl = document.getElementById('hap-figure');
  const hapImageEl = document.getElementById('hap-image');
  const hapImageLinkEl = document.getElementById('hap-image-link');
  const hapCaptionEl = document.getElementById('hap-caption');
  const HAP_COLLAPSE_KEY = 'genomeScanBrowser.hapPanelCollapsed.v1';

  // ---------------------------------------------------------------------
  // Formatting helpers
  // ---------------------------------------------------------------------
  function prettySpeciesName(key) {
    // "Bacteroides_ovatus_58035" -> "Bacteroides ovatus (58035)"
    const m = key.match(/^(.*)_(\d+)$/);
    if (!m) return key.replace(/_/g, ' ');
    return `${m[1].replace(/_/g, ' ')} (${m[2]})`;
  }

  // Rounds to `decimals` places but drops a trailing ".0"/".00" (e.g. "500.0"
  // -> "500") so whole-unit values like "500 kb" don't carry a fake-precise decimal.
  function trimTrailingZeros(v, decimals) {
    return v.toFixed(decimals).replace(/0+$/, '').replace(/\.$/, '');
  }

  function formatBp(bp) {
    const abs = Math.abs(bp);
    if (abs >= 1e6) return trimTrailingZeros(bp / 1e6, 2) + ' Mb';
    if (abs >= 1e3) return trimTrailingZeros(bp / 1e3, 1) + ' kb';
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

  async function loadPeaks() {
    try {
      const r = await fetch(`${DATA_DIR}/peaks.json`);
      if (!r.ok) return {};
      const data = await r.json();
      return (data && typeof data === 'object') ? data : {};
    } catch (e) {
      console.warn('Could not load curated peaks:', e);
      return {};
    }
  }

  function getPeaksFor(species) {
    return (species && peaksData[species]) || [];
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

  // A "sweep region" is one maximal run of >= `threshold` consecutive pooled
  // windows whose call is non-neutral (hard OR soft), then classified as hard
  // or soft by the majority call among the middle 50% of the run (first and
  // last quarter of the run dropped). Ties -> hard. Runs are defined over the
  // pooled colorCode that also colors each dot (getPooled), not the raw noisy
  // per-window CNN label, so the bands line up with the plotted signal.
  function classifyRun(pooled, start, endExclusive) {
    const runLen = endExclusive - start;
    const q = Math.floor(runLen / 4);           // drop first/last quarter
    let hard = 0, soft = 0;
    for (let i = start + q; i < endExclusive - q; i++) {
      if (pooled.colorCode[i] === LABEL_HARD) hard++;
      else if (pooled.colorCode[i] === LABEL_SOFT) soft++;
    }
    return hard >= soft ? 'hard' : 'soft';
  }

  function getSweepRuns(species, threshold) {
    const key = `${species}|${state.binSize}|${threshold}`;
    if (runsCache.has(key)) return runsCache.get(key);

    const entry = speciesCache.get(species);
    const pooled = getPooled(species);
    const runs = [];
    const pushRun = (start, endExclusive) => {
      const runLen = endExclusive - start;
      if (runLen < threshold) return;
      runs.push({
        startIdx: start,
        endIdx: endExclusive - 1,
        xStart: entry.position[start],
        xEnd: entry.position[endExclusive - 1],
        length: runLen,
        kind: classifyRun(pooled, start, endExclusive),
      });
    };
    for (const c of entry.manifest.contigs) {
      let runStart = -1;
      for (let i = c.start_index; i < c.end_index; i++) {
        if (pooled.colorCode[i] !== LABEL_NEUTRAL) {
          if (runStart === -1) runStart = i;
        } else if (runStart !== -1) {
          pushRun(runStart, i);
          runStart = -1;
        }
      }
      if (runStart !== -1) pushRun(runStart, c.end_index);
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
  const MARGIN = { top: 28, right: 20, bottom: 46, left: 60 };
  let renderScheduled = false;

  // Screen-space rects of the overlay boxes as last drawn on the live canvas,
  // for pointer hit-testing: [{kind, x, y, w, h}] in CSS pixels.
  let liveOverlayRects = [];

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

  // The two export toggles (caption / color key) double as a live preview: when
  // either is checked, the on-screen render draws the same overlay box it would
  // bake into the PNG, so the user can see and position it before exporting.
  function currentOverlayOptions() {
    const showCaption = exportCaptionToggle.checked;
    const showLegend = exportLegendToggle.checked;
    if (!showCaption && !showLegend) return null;
    return { showCaption, showLegend };
  }

  function render() {
    if (!state.species || !speciesCache.has(state.species)) return;
    const { w, h, dpr } = resizeCanvasToDisplaySize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    liveOverlayRects = renderCore(w, h, currentOverlayOptions()) || [];
  }

  // Draws the full chart into whatever `ctx` currently points at, using a
  // (w, h) viewport in CSS-pixel units. Shared by the live on-screen render
  // and the high-resolution PNG export (which temporarily swaps `ctx` to an
  // offscreen canvas scaled up by EXPORT_SCALE). `exportOptions` is null when
  // neither overlay is wanted; otherwise it's {showCaption, showLegend}, each
  // opt-in via a checkbox next to the export button. Those checkboxes now also
  // preview live on the canvas, so the overlay the user sees is exactly what
  // the PNG will contain.
  function renderCore(w, h, exportOptions) {
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
    if (state.annotateSweeps) {
      const runs = getSweepRuns(state.species, state.sweepThreshold);
      drawRuns(runs.filter(r => r.kind === 'hard'), colors.bandHard, colors.bandHardEdge, xMin, xMax, xToPx, plotH);
      drawRuns(runs.filter(r => r.kind === 'soft'), colors.bandSoft, colors.bandSoftEdge, xMin, xMax, xToPx, plotH);
    }

    // --- curated H12 peak bands + user gene/region annotations ---
    // Peaks carry the same {contigNum, startBp, endBp, text, color} shape as
    // user annotations, so both go through geneAnnotationPixels / drawGeneLabel.
    const bandSpecs = [
      ...(state.showPeaks ? getPeaksFor(state.species).map(p => ({ ...p, color: peakColor(p.kind) })) : []),
      ...getGeneAnnotationsFor(state.species),
    ];
    const genes = bandSpecs
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

    // Neutral dots vastly outnumber hard/soft ones, so they stay simple
    // filled squares (cheap to rasterize in bulk); hard/soft are drawn as
    // circles since there are few enough of them that the extra per-point
    // cost of an arc doesn't add up to a rendering slowdown.
    const colorFor = [colors.neutral, colors.hard, colors.soft];
    const squareSize = 1.6;
    const circleRadius = 1.45;
    for (let cc = 0; cc < 3; cc++) {
      if (!buckets[cc].length) continue;
      ctx.fillStyle = colorFor[cc];
      if (cc === LABEL_NEUTRAL) {
        const s = squareSize;
        for (const i of buckets[cc]) {
          const px = xToPx(entry.position[i]);
          const py = yToPx(pooled.y[i]);
          ctx.fillRect(px - s / 2, py - s / 2, s, s);
        }
      } else {
        const r = circleRadius;
        ctx.beginPath();
        for (const i of buckets[cc]) {
          const px = xToPx(entry.position[i]);
          const py = yToPx(pooled.y[i]);
          ctx.moveTo(px + r, py);
          ctx.arc(px, py, r, 0, Math.PI * 2);
        }
        ctx.fill();
      }
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

    const overlayRects = exportOptions
      ? drawExportOverlays(colors, w, h, plotW, exportOptions)
      : [];

    updateInfoPanels(entry);
    return overlayRects;
  }

  // Shared box chrome (background + border) for the export-only overlays
  // below, positioned/sized by the caller.
  function drawExportBoxChrome(colors, x0, y0, boxW, boxH) {
    ctx.fillStyle = hexToRgba(colors.surface1, 0.92);
    ctx.fillRect(x0, y0, boxW, boxH);
    ctx.strokeStyle = colors.baseline;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, boxW - 1, boxH - 1);
  }

  // Places the opt-in export overlays (caption / color key) and draws them.
  // Default spot is the top-right of the plot, caption stacked above legend;
  // once the user drags a box on the canvas it gets an explicit {fx,fy} in
  // state.overlayPos and is placed there instead (always clamped to stay
  // fully on-canvas). Returns [{kind,x,y,w,h}] for the boxes actually drawn,
  // in the (w,h) coordinate space, so the pointer handlers can hit-test them
  // and the PNG export lands each box exactly where the user left it.
  function drawExportOverlays(colors, w, h, plotW, exportOptions) {
    const boxes = [];
    if (exportOptions.showCaption) boxes.push(measureExportCaptionBox());
    if (exportOptions.showLegend) boxes.push(measureExportLegendBox(colors));

    const defaultRight = MARGIN.left + plotW - 10;
    let defaultY = MARGIN.top + 10;
    const rects = [];

    for (const box of boxes) {
      const pos = state.overlayPos[box.kind];
      let x0 = pos ? pos.fx * w : defaultRight - box.boxW;
      let y0 = pos ? pos.fy * h : defaultY;
      x0 = Math.max(0, Math.min(w - box.boxW, x0));
      y0 = Math.max(0, Math.min(h - box.boxH, y0));

      if (box.kind === 'caption') drawExportCaptionBox(colors, box, x0, y0);
      else drawExportLegendBox(colors, box, x0, y0);

      rects.push({ kind: box.kind, x: x0, y: y0, w: box.boxW, h: box.boxH });
      if (!pos) defaultY = y0 + box.boxH + 8;
    }
    return rects;
  }

  // In-image caption for PNG exports: pooling window size always, plus each
  // run-length threshold only when its highlighting is actually turned on
  // (an unused threshold value would be misleading to someone who only sees
  // the flat PNG and can't tell the checkbox was off).
  function measureExportCaptionBox() {
    const lines = [`Pooled: ${state.binSize} window${state.binSize === 1 ? '' : 's'}`];
    if (state.annotateSweeps) lines.push(`Sweep region: ≥ ${state.sweepThreshold} consecutive sweep calls (hard/soft by majority of middle 50%)`);

    ctx.font = '11px system-ui, sans-serif';
    const padX = 9, padY = 7, lineH = 15;
    let textW = 0;
    for (const line of lines) textW = Math.max(textW, ctx.measureText(line).width);
    return {
      kind: 'caption', lines, padX, padY, lineH,
      boxW: textW + padX * 2,
      boxH: lines.length * lineH + padY * 2,
    };
  }

  function drawExportCaptionBox(colors, box, x0, y0) {
    drawExportBoxChrome(colors, x0, y0, box.boxW, box.boxH);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = colors.textSecondary;
    box.lines.forEach((line, i) => {
      ctx.fillText(line, x0 + box.padX, y0 + box.padY + i * box.lineH);
    });
  }

  // In-image color key for PNG exports, mirroring the topbar legend
  // (including the run-region swatches, shown only when that highlighting
  // is actually on).
  function measureExportLegendBox(colors) {
    const items = [
      { text: 'Neutral', swatchFill: colors.neutral },
      { text: 'Hard sweep', swatchFill: colors.hard },
      { text: 'Soft sweep', swatchFill: colors.soft },
    ];
    if (state.annotateSweeps && state.species && speciesCache.has(state.species)) {
      const runs = getSweepRuns(state.species, state.sweepThreshold);
      if (runs.some(r => r.kind === 'hard')) items.push({ text: 'Hard-sweep region', swatchFill: colors.bandHard, swatchStroke: colors.bandHardEdge });
      if (runs.some(r => r.kind === 'soft')) items.push({ text: 'Soft-sweep region', swatchFill: colors.bandSoft, swatchStroke: colors.bandSoftEdge });
    }
    if (state.showPeaks) {
      const peaks = getPeaksFor(state.species);
      if (peaks.some(p => p.kind === 'hard')) {
        items.push({ text: 'H12 peak (hard)', swatchFill: hexToRgba(PEAK_COLOR_HARD, 0.16), swatchStroke: hexToRgba(PEAK_COLOR_HARD, 0.7) });
      }
      if (peaks.some(p => p.kind === 'soft')) {
        items.push({ text: 'H12 peak (soft)', swatchFill: hexToRgba(PEAK_COLOR_SOFT, 0.16), swatchStroke: hexToRgba(PEAK_COLOR_SOFT, 0.7) });
      }
    }

    ctx.font = '11px system-ui, sans-serif';
    const padX = 9, padY = 7, lineH = 16, swatchSize = 10, swatchGap = 7;
    let textW = 0;
    for (const it of items) textW = Math.max(textW, ctx.measureText(it.text).width);
    return {
      kind: 'legend', items, padX, padY, lineH, swatchSize, swatchGap,
      boxW: swatchSize + swatchGap + textW + padX * 2,
      boxH: items.length * lineH + padY * 2,
    };
  }

  function drawExportLegendBox(colors, box, x0, y0) {
    drawExportBoxChrome(colors, x0, y0, box.boxW, box.boxH);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    box.items.forEach((it, i) => {
      const rowCy = y0 + box.padY + i * box.lineH + box.lineH / 2;
      const sx = x0 + box.padX;
      ctx.fillStyle = it.swatchFill;
      ctx.fillRect(sx, rowCy - box.swatchSize / 2, box.swatchSize, box.swatchSize);
      if (it.swatchStroke) {
        ctx.strokeStyle = it.swatchStroke;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(sx + 0.5, rowCy - box.swatchSize / 2 + 0.5, box.swatchSize - 1, box.swatchSize - 1);
      }
      ctx.fillStyle = colors.textSecondary;
      ctx.fillText(it.text, sx + box.swatchSize + box.swatchGap, rowCy);
    });
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

    // per-contig bp ticks, only for contigs wide enough on screen. Contigs
    // are visited left-to-right (manifest order matches x_start order), so a
    // single running "next allowed x" cursor across the whole pass is enough
    // to guarantee no two tick labels overlap -- both within one wide contig
    // and across a boundary between two narrow adjacent contigs, which is
    // the case that produced overlapping labels when fully zoomed out.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = colors.textMuted;
    ctx.beginPath();
    let nextAllowedLabelPx = -Infinity;
    const LABEL_GAP_PX = 10;
    visible.forEach((c) => {
      const segX0 = xToPx(Math.max(c.x_start, xMin));
      const segX1 = xToPx(Math.min(c.x_end, xMax));
      const segW = segX1 - segX0;
      if (segW < 90) return;

      const bpLo = Math.max(0, xMin - c.x_start);
      const bpHi = Math.min(c.x_end - c.x_start, xMax - c.x_start);
      const bpRange = bpHi - bpLo;
      if (bpRange <= 0) return;
      // Divisor of 130 (vs. a tighter default) deliberately asks for coarser,
      // more widely-spaced ticks -- fewer labels to begin with, especially
      // when zoomed all the way out and many contigs are on screen at once.
      const targetTicks = Math.max(2, Math.floor(segW / 130));
      const step = niceTickStep(bpRange, targetTicks);
      const firstTick = Math.ceil(bpLo / step) * step;
      for (let bp = firstTick; bp <= bpHi; bp += step) {
        const cx = c.x_start + bp;
        if (cx < xMin || cx > xMax) continue;
        const px = xToPx(cx);
        const label = formatBp(bp);
        const halfW = ctx.measureText(label).width / 2;
        if (px - halfW < nextAllowedLabelPx) continue;
        ctx.moveTo(px, plotBottom);
        ctx.lineTo(px, plotBottom + 5);
        ctx.fillText(label, px, plotBottom + 8);
        nextAllowedLabelPx = px + halfW + LABEL_GAP_PX;
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
      setPeakControlsState();
      setLegendRunVisibility();
      state.hapSites = await loadHapSites(species);
      state.hapSelected = null;
      renderHapPanel();
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
    setLegendRunVisibility();   // pooling changes which runs (and their hard/soft split) exist
    scheduleRender();
  }
  poolSlider.addEventListener('input', () => setBinSize(+poolSlider.value));
  poolNumber.addEventListener('change', () => setBinSize(+poolNumber.value));

  // The topbar's hard/soft-sweep-region chips show only for the classes that
  // are actually on the plot right now: sweep highlighting on, and at least
  // one run classified that way for this species at this pooling.
  function setLegendRunVisibility() {
    let hasHard = false, hasSoft = false;
    if (state.annotateSweeps && state.species && speciesCache.has(state.species)) {
      const runs = getSweepRuns(state.species, state.sweepThreshold);
      hasHard = runs.some(r => r.kind === 'hard');
      hasSoft = runs.some(r => r.kind === 'soft');
    }
    legendHardRunEl.classList.toggle('hidden', !hasHard);
    legendSoftRunEl.classList.toggle('hidden', !hasSoft);
    // Only-soft-shown is the one case with an awkward gap (the hidden hard-run
    // slot would sit between "Soft sweep" and "Soft-sweep region"), so swap the
    // pair's order then.
    topbarLegendEl.classList.toggle('swap-run-order', hasSoft && !hasHard);
  }

  // Reflects the curated-peak toggle + current species into the sidebar note,
  // the checkbox enabled state, and the topbar legend chip.
  function setPeakControlsState() {
    const peaks = getPeaksFor(state.species);
    const n = peaks.length;
    showPeaksCb.disabled = n === 0;
    if (!n) {
      peakCountNoteEl.textContent = state.species ? 'No curated peaks for this species.' : '';
    } else {
      peakCountNoteEl.textContent = `${n} curated peak${n === 1 ? '' : 's'}`;
    }
    const on = state.showPeaks && n > 0;
    if (legendPeakHardEl) legendPeakHardEl.classList.toggle('hidden', !(on && peaks.some(p => p.kind === 'hard')));
    if (legendPeakSoftEl) legendPeakSoftEl.classList.toggle('hidden', !(on && peaks.some(p => p.kind === 'soft')));
  }

  annotateSweepsCb.addEventListener('change', () => { state.annotateSweeps = annotateSweepsCb.checked; setLegendRunVisibility(); scheduleRender(); });
  sweepThresholdInput.addEventListener('change', () => {
    state.sweepThreshold = Math.max(2, Math.round(+sweepThresholdInput.value) || 2);
    sweepThresholdInput.value = state.sweepThreshold;
    setLegendRunVisibility();
    scheduleRender();
  });
  showPeaksCb.addEventListener('change', () => {
    state.showPeaks = showPeaksCb.checked;
    setPeakControlsState();
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

    const exportOptions = currentOverlayOptions();

    const liveCtx = ctx;
    ctx = offCtx;
    try {
      renderCore(w, h, exportOptions);
    } finally {
      ctx = liveCtx;
      scheduleRender();   // repaint the live canvas at its own resolution
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
  // Re-render on toggle so the caption / color-key box appears (or disappears)
  // on the live canvas immediately, not only in the exported PNG.
  exportCaptionToggle.addEventListener('change', scheduleRender);
  exportLegendToggle.addEventListener('change', scheduleRender);

  // Initialize control DOM from state defaults.
  setBinSize(state.binSize);
  annotateSweepsCb.checked = state.annotateSweeps;
  sweepThresholdInput.value = state.sweepThreshold;
  showPeaksCb.checked = state.showPeaks;
  setLegendRunVisibility();
  setPeakControlsState();

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
  // Active drag of a caption/legend overlay box, or null.
  // {kind, startClientX, startClientY, boxX0, boxY0, canvasW, canvasH}
  let overlayDrag = null;

  // Topmost overlay box under (px, py) — CSS pixels relative to the canvas —
  // or null. Iterated back-to-front so the last-drawn box wins on overlap.
  function overlayHitTest(px, py) {
    for (let i = liveOverlayRects.length - 1; i >= 0; i--) {
      const r = liveOverlayRects[i];
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return r;
    }
    return null;
  }

  canvas.addEventListener('pointerdown', (e) => {
    if (!state.species) return;
    const rect = canvas.getBoundingClientRect();
    const hit = overlayHitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (hit) {
      overlayDrag = {
        kind: hit.kind,
        startClientX: e.clientX,
        startClientY: e.clientY,
        boxX0: hit.x,
        boxY0: hit.y,
        canvasW: rect.width,
        canvasH: rect.height,
      };
      canvas.style.cursor = 'grabbing';
      hideTooltip();
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    dragging = true;
    dragLastX = e.clientX;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (overlayDrag) {
      let dx = e.clientX - overlayDrag.startClientX;
      let dy = e.clientY - overlayDrag.startClientY;
      // Shift locks the move to whichever axis the drag has favored so far,
      // recomputed from the drag origin each move so toggling Shift mid-drag
      // stays predictable.
      if (e.shiftKey) {
        if (Math.abs(dx) >= Math.abs(dy)) dy = 0; else dx = 0;
      }
      state.overlayPos[overlayDrag.kind] = {
        fx: (overlayDrag.boxX0 + dx) / overlayDrag.canvasW,
        fy: (overlayDrag.boxY0 + dy) / overlayDrag.canvasH,
      };
      scheduleRender();
      return;
    }
    if (dragging) {
      const dx = e.clientX - dragLastX;
      dragLastX = e.clientX;
      const scale = pixelToDataScale();
      panBy(-dx * scale);
      scheduleRender();
      hideTooltip();
    } else {
      const rect = canvas.getBoundingClientRect();
      if (overlayHitTest(e.clientX - rect.left, e.clientY - rect.top)) {
        canvas.style.cursor = 'move';
        hideTooltip();
      } else {
        canvas.style.cursor = '';
        showTooltip(e);
      }
    }
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    overlayDrag = null;
    canvas.style.cursor = '';
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerleave', () => {
    dragging = false;
    overlayDrag = null;
    canvas.style.cursor = '';
    hideTooltip();
  });

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
    if (!state.species || dragging || overlayDrag) return;
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
    if (state.annotateSweeps) {
      const run = getSweepRuns(state.species, state.sweepThreshold)
        .find(r => best >= r.startIdx && best <= r.endIdx);
      if (run) lines.push(`▸ ${run.kind === 'hard' ? 'Hard' : 'Soft'}-sweep region (${run.length} windows)`);
    }
    if (state.showPeaks) {
      for (const p of getPeaksFor(state.species)) {
        if (p.contigNum === contig.contig_num && bpLocal >= p.startBp && bpLocal <= p.endBp) {
          lines.push(`▸ ${p.text} (${formatBp(p.startBp)}–${formatBp(p.endBp)})`);
        }
      }
    }
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
  // The plot canvas fills .plot-canvas-wrap, which shrinks/grows as the
  // haplotype panel below it opens, loads an image, or is collapsed. Re-render
  // (and thus re-size the canvas bitmap) on any such change.
  if (window.ResizeObserver && canvasWrapEl) {
    new ResizeObserver(() => scheduleRender()).observe(canvasWrapEl);
  }

  // ---------------------------------------------------------------------
  // Haplotype snapshots: a small curated set of pre-rendered haplotype
  // images per species (1/4, 1/2, 3/4 of the genome + the longest hard-run
  // and soft-run windows), reached from the "Haplotype snapshots" panel
  // below the scan. Absent-file => feature silently hidden for that species.
  // ---------------------------------------------------------------------
  const HAP_TYPE_ORDER = { baseline: 0, hard: 1, soft: 2 };

  async function loadHapSites(species) {
    try {
      const r = await fetch(`${DATA_DIR}/${species}_haplotype_sites.json`);
      if (!r.ok) return null;
      const data = await r.json();
      if (!data || !Array.isArray(data.sites) || !data.sites.length) return null;
      data.sites.sort((a, b) =>
        (HAP_TYPE_ORDER[a.type] - HAP_TYPE_ORDER[b.type]) || (a.x - b.x));
      return data;
    } catch (e) {
      return null;
    }
  }

  function hapSiteLabel(s) {
    if (s.type === 'baseline') return s.label;
    const mb = (s.bp_lo / 1e6).toFixed(2);
    return `${s.type === 'hard' ? 'Hard' : 'Soft'} run · ${mb} Mb`;
  }

  function renderHapPanel() {
    hapChipsEl.innerHTML = '';
    hapFigureEl.hidden = true;
    hapImageEl.removeAttribute('src');
    state.hapSelected = null;

    if (!state.hapSites) {
      hapPanelEl.hidden = true;
      return;
    }
    hapPanelEl.hidden = false;

    const sites = state.hapSites.sites;
    const nHard = sites.filter(s => s.type === 'hard').length;
    const nSoft = sites.filter(s => s.type === 'soft').length;
    const nBase = sites.length - nHard - nSoft;
    hapPanelNoteEl.textContent =
      `${sites.length} windows` +
      (nBase ? ` · ${nBase} neutral` : '') +
      (nHard ? ` · ${nHard} hard-sweep run${nHard === 1 ? '' : 's'}` : '') +
      (nSoft ? ` · ${nSoft} soft-sweep run${nSoft === 1 ? '' : 's'}` : '');

    for (const s of sites) {
      const chip = document.createElement('button');
      chip.className = 'hap-chip';
      chip.dataset.key = s.key;
      const dot = document.createElement('span');
      dot.className = `dot type-${s.type}`;
      chip.appendChild(dot);
      chip.appendChild(document.createTextNode(hapSiteLabel(s)));
      chip.addEventListener('click', () => selectHapSite(s.key));
      hapChipsEl.appendChild(chip);
    }

    // restore collapsed preference
    let collapsed = false;
    try { collapsed = localStorage.getItem(HAP_COLLAPSE_KEY) === '1'; } catch (e) {}
    hapPanelEl.classList.toggle('is-collapsed', collapsed);
    hapToggleBtn.setAttribute('aria-expanded', String(!collapsed));
  }

  function selectHapSite(key) {
    const site = state.hapSites && state.hapSites.sites.find(s => s.key === key);
    if (!site) return;
    state.hapSelected = key;

    for (const chip of hapChipsEl.children) {
      chip.classList.toggle('is-active', chip.dataset.key === key);
    }

    const loc = (state.hapSites.sites.some(s => s.contig !== site.contig) ? `contig ${site.contig} · ` : '') +
      `${(site.bp_lo / 1e6).toFixed(3)}–${(site.bp_hi / 1e6).toFixed(3)} Mb`;
    hapImageEl.src = site.image;
    hapImageLinkEl.href = site.image;
    hapImageEl.alt = `${state.hapSites.pretty} haplotypes at ${loc} (${site.label})`;
    hapCaptionEl.textContent =
      `${state.hapSites.pretty} — ${site.label}. ${loc}. Pooled call here: ${site.call_here}. ` +
      `${state.hapSites.target_samples} genomes × ${state.hapSites.window_h} sites. ` +
      `Left: allele state. Right: mutation type.`;
    hapFigureEl.hidden = false;
    hapPanelEl.scrollTop = 0;   // keep the chips/header in view

    if (hapPanelEl.classList.contains('is-collapsed')) setHapCollapsed(false);

    // The panel just grew (image shown) — re-render so the canvas resizes to
    // the smaller area instead of overflowing and scrolling the page.
    scheduleRender();
  }

  function setHapCollapsed(collapsed) {
    hapPanelEl.classList.toggle('is-collapsed', collapsed);
    hapToggleBtn.setAttribute('aria-expanded', String(!collapsed));
    try { localStorage.setItem(HAP_COLLAPSE_KEY, collapsed ? '1' : '0'); } catch (e) {}
    scheduleRender();
  }

  hapToggleBtn.addEventListener('click', () => {
    setHapCollapsed(!hapPanelEl.classList.contains('is-collapsed'));
  });

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  (async function init() {
    const [, peaks] = await Promise.all([loadSpeciesIndex(), loadPeaks()]);
    peaksData = peaks;
    if (!speciesIndex.length) return;
    const params = new URLSearchParams(location.search);
    if (params.get('peaks') === '1') {
      state.showPeaks = true;
      showPeaksCb.checked = true;
    }
    const want = params.get('species');
    const start = (want && speciesIndex.some(s => s.species === want))
      ? want : speciesIndex[0].species;
    await selectSpecies(start);

    const wantHap = params.get('hap');
    if (wantHap && state.hapSites && state.hapSites.sites.some(s => s.key === wantHap)) {
      requestAnimationFrame(() => selectHapSite(wantHap));
    }
  })();
})();
