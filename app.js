/* MNIST Dense-CA (50 steps) browser demo using ONNX Runtime Web (WASM).
 *
 * Files expected:
 *   assets/model.onnx  (exported from your .pt)
 *   assets/means.json  (10 x 784 float array)
 */

'use strict';

const MODEL_URL = 'assets/model.onnx';
const MEANS_URL = 'assets/means.json';

const NUM_BLOCKS = 50;
const SHOW_MEAN_EVERY = 5;

// Thumbnails strip settings
const STRIP_HEIGHT = 48;   // CSS pixels
const STRIP_PAD = 2;

// Softmax temperature for confidence (higher = sharper)
const SOFTMAX_TEMP = 12.0;

let session = null;
let means = null; // Array<Float32Array> length 10, each length 784

let frames = null; // Array<Float32Array> length 50
let frameIdx = 0;

let animTimer = null;
let lastScoreText = ''; // keep last confidence text (no flicker)

function $(id) { return document.getElementById(id); }

function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

function setStatus(msg) { $('status').textContent = msg; }

function setVerdict(html) { $('verdict').innerHTML = html; }

function float28ToImageData(f32) {
  const w = 28, h = 28;
  const img = new ImageData(w, h);
  const d = img.data;
  for (let i = 0; i < w * h; i++) {
    const v = clamp01(f32[i]);
    const b = (v * 255) | 0;
    const j = i * 4;
    d[j + 0] = b;
    d[j + 1] = b;
    d[j + 2] = b;
    d[j + 3] = 255;
  }
  return img;
}

function draw28ToCanvas(f32, canvas, scale = 10) {
  const tmp = document.createElement('canvas');
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext('2d', { willReadFrequently: false });
  tctx.putImageData(float28ToImageData(f32), 0, 0);

  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const w = 28 * scale, h = 28 * scale;
  const ox = ((canvas.width - w) / 2) | 0;
  const oy = ((canvas.height - h) / 2) | 0;
  ctx.drawImage(tmp, 0, 0, 28, 28, ox, oy, w, h);
}
function preprocessCanvasTo28(drawCanvas) {
  // 1) Find ink bbox on the original draw canvas
  const W = drawCanvas.width, H = drawCanvas.height;
  const dctx = drawCanvas.getContext('2d', { willReadFrequently: true });
  const data = dctx.getImageData(0, 0, W, H).data;

  const TH = 0.10; // ink threshold in [0..1] (tune if needed)
  let minX = W, minY = H, maxX = -1, maxY = -1;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const r = data[i + 0], g = data[i + 1], b = data[i + 2];
      const a = data[i + 3] / 255.0;
      const v = (Math.max(r, g, b) / 255.0) * a; // bright ink on black
      if (v > TH) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  // 2) Convert bbox -> square crop with padding
  let sx = 0, sy = 0, side = Math.min(W, H);

  if (maxX >= 0) {
    const bw = (maxX - minX + 1);
    const bh = (maxY - minY + 1);
    const pad = Math.floor(Math.max(bw, bh) * 0.20) + 4; // padding
    side = Math.max(bw, bh) + 2 * pad;
    side = Math.min(side, Math.min(W, H));

    const cx = (minX + maxX + 1) * 0.5;
    const cy = (minY + maxY + 1) * 0.5;

    sx = Math.round(cx - side * 0.5);
    sy = Math.round(cy - side * 0.5);

    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx + side > W) sx = W - side;
    if (sy + side > H) sy = H - side;
  }

  // 3) Downscale that square crop to 28×28
  const tmp = document.createElement('canvas');
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });

  tctx.imageSmoothingEnabled = true;
  tctx.clearRect(0, 0, 28, 28);
  tctx.drawImage(drawCanvas, sx, sy, side, side, 0, 0, 28, 28);

  // 4) Convert to Float32 [0..1] (preference to marker color)
  const img = tctx.getImageData(0, 0, 28, 28).data;
  const out = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = img[i * 4 + 0];
    const g = img[i * 4 + 1];
    const b = img[i * 4 + 2];
    const a = img[i * 4 + 3] / 255.0;
    out[i] = clamp01((Math.max(r, g, b) / 255.0) * a);
  }
  return out;
}


function corr01(x, y) {
  // cosine similarity in [0,1] for nonnegative images
  let dot = 0, nx = 0, ny = 0;
  for (let i = 0; i < 784; i++) {
    const a = x[i], b = y[i];
    dot += a * b;
    nx  += a * a;
    ny  += b * b;
  }
  const sim = dot / (Math.sqrt(nx * ny) + 1e-12);
  return clamp01(sim);
}


function softmaxFromScores(scores, temp = SOFTMAX_TEMP) {
  let maxL = -Infinity;
  const logits = new Float32Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    const l = scores[i] * temp;
    logits[i] = l;
    if (l > maxL) maxL = l;
  }
  let sum = 0;
  const exps = new Float32Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    const e = Math.exp(logits[i] - maxL);
    exps[i] = e;
    sum += e;
  }
  const probs = new Float32Array(scores.length);
  for (let i = 0; i < scores.length; i++) probs[i] = exps[i] / (sum + 1e-12);
  return probs;
}

function probsFromImage(img784) {
  // "prob" = correlation directly (not normalized to sum=1)
  const probs = new Float32Array(10);

  let bestD = 0;
  let bestP = -Infinity;

  for (let d = 0; d < 10; d++) {
    const c = corr01(img784, means[d]);   // in [0,1]
    probs[d] = c;
    if (c > bestP) { bestP = c; bestD = d; }
  }

  return { probs, bestD, bestP };
}


function pctInt(p) { return Math.max(0, Math.min(100, Math.round(p * 100))); }

function probsString(probs) {
  const parts = [];
  for (let d = 0; d < 10; d++) parts.push(`${d}:${pctInt(probs[d])}%`);
  return parts.join(' ');
}

function resizeCanvasToDisplaySize(canvas, desiredCssHeight = null) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const hCss = desiredCssHeight !== null ? desiredCssHeight : canvas.clientHeight;
  const h = Math.max(1, Math.floor(hCss * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

function renderStrip(highlightIdx = -1) {
  const canvas = $('grid');
  resizeCanvasToDisplaySize(canvas, STRIP_HEIGHT);

  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const dpr = window.devicePixelRatio || 1;
  const pad = Math.floor(STRIP_PAD * dpr);

  const cell = Math.max(1, Math.floor((canvas.width - 2 * pad) / NUM_BLOCKS));
  const usableW = cell * NUM_BLOCKS;
  const ox = ((canvas.width - usableW) / 2) | 0;
  const oy = pad;

  const tmp = document.createElement('canvas');
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext('2d');

  for (let i = 0; i < NUM_BLOCKS; i++) {
    const x = ox + i * cell;
    tctx.putImageData(float28ToImageData(frames[i]), 0, 0);
    ctx.drawImage(tmp, 0, 0, 28, 28, x, oy, cell, canvas.height - 2 * oy);

    if (((i + 1) % SHOW_MEAN_EVERY) === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.fillRect(x + cell - 1, oy, 1, canvas.height - 2 * oy);
    }

    if (i === highlightIdx) {
      ctx.strokeStyle = '#00e1ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, oy + 1, cell - 2, canvas.height - 2 * oy - 2);
    }
  }
}

function setFrame(i) {
  if (!frames) return;
  frameIdx = Math.max(0, Math.min(NUM_BLOCKS - 1, i));

  draw28ToCanvas(frames[frameIdx], $('animOut'), 10);

  const step = frameIdx + 1;

  // Update score only every 5 steps; keep the old one otherwise (no flicker)
  if ((step % SHOW_MEAN_EVERY) === 0) {
    const { bestD, bestP } = probsFromImage(frames[frameIdx]);
    draw28ToCanvas(means[bestD], $('animMean'), 10);
    lastScoreText = `  best=${bestD}  conf=${pctInt(bestP)}%`;
  }

  $('frameLabel').textContent = `frame: ${step}/${NUM_BLOCKS}` + lastScoreText;
  renderStrip(frameIdx);
}

function stopAnim() {
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
}

function startAnim() {
  stopAnim();
  const speedMs = parseInt($('speed').value, 10);
  animTimer = setInterval(() => {
    if (!frames) return;
    setFrame((frameIdx + 1) % NUM_BLOCKS);
  }, speedMs);
}

async function runModel(input784) {
  if (!session) throw new Error('Session not loaded');
  const x = new ort.Tensor('float32', input784, [1, 1, 28, 28]);
  const out = await session.run({ x });

  const arr = new Array(NUM_BLOCKS);
  for (let i = 0; i < NUM_BLOCKS; i++) {
    const k = `out${String(i).padStart(2, '0')}`;
    const t = out[k];
    if (!t) throw new Error(`Missing output '${k}'. Re-export ONNX with tools/export_onnx.py.`);
    arr[i] = new Float32Array(t.data.slice(0, 784));
  }
  return arr;
}

function setupDrawing() {
  const canvas = $('draw');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  // Allow scroll/zoom by default; disable only during an active stroke.
  const DEFAULT_TOUCH_ACTION = 'pan-x pan-y pinch-zoom';
  canvas.style.touchAction = DEFAULT_TOUCH_ACTION;

  function clear() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  clear();

  let drawing = false;

  function brush() {
    const w = parseInt($('brushSize').value, 10);
    const col = $('brushColor').value || '#ffffff';
    ctx.lineWidth = w;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = col;
  }

  function pos(evt) {
    const r = canvas.getBoundingClientRect();
    return {
      x: (evt.clientX - r.left) * (canvas.width / r.width),
      y: (evt.clientY - r.top) * (canvas.height / r.height),
    };
  }

  canvas.addEventListener('pointerdown', (e) => {
    drawing = true;

    // Lock gestures while drawing so finger-drag produces continuous strokes.
    canvas.style.touchAction = 'none';
    e.preventDefault();

    brush();
    const p = pos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);

    canvas.setPointerCapture(e.pointerId);
  }, { passive: false });

  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    e.preventDefault();

    const p = pos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }, { passive: false });

  function endStroke(e) {
    if (!drawing) return;
    drawing = false;

    // Restore scroll/zoom immediately after the stroke ends.
    canvas.style.touchAction = DEFAULT_TOUCH_ACTION;

    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  }

  canvas.addEventListener('pointerup', endStroke, { passive: false });
  canvas.addEventListener('pointercancel', endStroke, { passive: false });

  $('clearBtn').addEventListener('click', () => {
    clear();
    frames = null;
    lastScoreText = '';
    stopAnim();
    $('frameLabel').textContent = 'frame: -';
    $('inputInfo').textContent = '';
    setVerdict('');

    $('grid').getContext('2d').clearRect(0, 0, $('grid').width, $('grid').height);
    $('animOut').getContext('2d').clearRect(0, 0, $('animOut').width, $('animOut').height);
    $('animMean').getContext('2d').clearRect(0, 0, $('animMean').width, $('animMean').height);
    $('inputPreview').getContext('2d').clearRect(0, 0, $('inputPreview').width, $('inputPreview').height);

    // Also restore touch-action if user clears mid-stroke
    drawing = false;
    canvas.style.touchAction = DEFAULT_TOUCH_ACTION;
  });
}

async function main() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';

  setupDrawing();

  $('brushSize').addEventListener('input', () => {
    $('brushVal').textContent = $('brushSize').value;
  });

  $('speed').addEventListener('input', () => {
    $('speedVal').textContent = `${$('speed').value}ms`;
    if (animTimer) startAnim();
  });

  $('playBtn').addEventListener('click', startAnim);
  $('pauseBtn').addEventListener('click', stopAnim);

  $('prevBtn').addEventListener('click', () => {
    if (!frames) return;
    stopAnim();
    setFrame((frameIdx - 1 + NUM_BLOCKS) % NUM_BLOCKS);
  });

  $('nextBtn').addEventListener('click', () => {
    if (!frames) return;
    stopAnim();
    setFrame((frameIdx + 1) % NUM_BLOCKS);
  });

  $('grid').addEventListener('click', (e) => {
    if (!frames) return;
    const canvas = $('grid');
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (canvas.width / r.width);

    const dpr = window.devicePixelRatio || 1;
    const pad = Math.floor(STRIP_PAD * dpr);
    const cell = Math.max(1, Math.floor((canvas.width - 2 * pad) / NUM_BLOCKS));
    const usableW = cell * NUM_BLOCKS;
    const ox = ((canvas.width - usableW) / 2) | 0;

    const i = Math.floor((x - ox) / cell);
    if (i >= 0 && i < NUM_BLOCKS) {
      stopAnim();
      setFrame(i);
    }
  });

  window.addEventListener('resize', () => {
    if (frames) renderStrip(frameIdx);
  });

  setStatus('Loading means.json …');
  const meansJson = await (await fetch(MEANS_URL)).json();
  means = meansJson.map(row => Float32Array.from(row));
  if (means.length !== 10 || means[0].length !== 784) {
    throw new Error('means.json must be shape [10][784]. Recreate via tools/npy_to_json.py');
  }

  setStatus('Loading model.onnx …');
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });

  // warm-up: compile/initialize kernels now, not on first Run
  setStatus('Warming up model …');
  const warm = new ort.Tensor('float32', new Float32Array(28 * 28), [1, 1, 28, 28]);
  await session.run({ x: warm });
  
  setStatus('Ready.');

  $('speedVal').textContent = `${$('speed').value}ms`;

  $('runBtn').addEventListener('click', async () => {
    try {
      stopAnim();
      setStatus('Preprocessing …');

      const input784 = preprocessCanvasTo28($('draw'));
      draw28ToCanvas(input784, $('inputPreview'), 5);

      const inp = probsFromImage(input784);
      $('inputInfo').textContent =
        `input: best=${inp.bestD} conf=${pctInt(inp.bestP)}%  |  ` + probsString(inp.probs);

      setStatus('Running model …');
      frames = await runModel(input784);

      lastScoreText = '';
      renderStrip(-1);
      setFrame(0);

      const final = probsFromImage(frames[NUM_BLOCKS - 1]);

      const preds = [];
      for (let k = SHOW_MEAN_EVERY; k <= NUM_BLOCKS; k += SHOW_MEAN_EVERY) {
        const p = probsFromImage(frames[k - 1]);
        preds.push(`${String(k).padStart(2,'0')}:${p.bestD}(${pctInt(p.bestP)}%)`);
      }

      setVerdict(
        `<div class="mono">final: digit=<b>${final.bestD}</b>  confidence=<b>${pctInt(final.bestP)}%</b></div>` +
        `<div class="mono" style="margin-top:8px;">every ${SHOW_MEAN_EVERY}: ${preds.join(' | ')}</div>`
      );

      setStatus('Done.');
      startAnim(); // start animating immediately after Run
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message || String(err)}`);
    }
  });
}

main().catch(err => {
  console.error(err);
  setStatus(`Fatal: ${err.message || String(err)}`);
});
