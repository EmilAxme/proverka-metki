// Проверка скрытой метки TrustMark (вариант P) с номером свидетельства COPYTRUST.
// Повторяет tools/onnx_read.py: зеркало нет/да → центральный кусок 100..70% → центральный квадрат →
// сжатие до 224 как PIL BILINEAR → ONNX → BCH_5 (код Adobe) → контрольный символ.
'use strict';

const MODEL_URL = 'model/decoder_P.onnx';
// GitHub Pages отдаёт модель сжатой: content-length там меньше распакованных байт, и проценты уходили за 100
const MODEL_BYTES = 23891025;
const CROPS = [1.0, 0.95, 0.9, 0.86, 0.8, 0.75, 0.7];
const ABC = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const SIZE = 224;

ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
ort.env.wasm.numThreads = 1;

let sessionPromise = null;
const listeners = new Set();

// одна загрузка на всех: прогресс получают и строка статуса, и каждая карточка, которая ждёт модель
function loadModel(onProgress) {
  if (onProgress) listeners.add(onProgress);
  const p = startLoad();
  p.then(() => listeners.delete(onProgress), () => listeners.delete(onProgress));
  return p;
}

function startLoad() {
  const onProgress = v => listeners.forEach(f => f(v));
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const r = await fetch(MODEL_URL);
      if (!r.ok) throw new Error('модель не загрузилась: ' + r.status);
      const len = Number(r.headers.get('content-length')) || 0;
      const total = !r.headers.get('content-encoding') && len ? len : MODEL_BYTES;
      const reader = r.body.getReader();
      const parts = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        got += value.length;
        onProgress(Math.min(got / total, 0.99));
      }
      const buf = new Uint8Array(got);
      let off = 0;
      for (const p of parts) { buf.set(p, off); off += p.length; }
      return ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    })();
    sessionPromise.catch(() => { sessionPromise = null; });
  }
  return sessionPromise;
}

// --- картинка как {w, h, d: Uint8Array RGB} ---

async function decodeFile(file) {
  const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
  const d = new Uint8Array(c.width * c.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { d[j] = rgba[i]; d[j + 1] = rgba[i + 1]; d[j + 2] = rgba[i + 2]; }
  bmp.close && bmp.close();
  return { w: c.width, h: c.height, d };
}

function crop(im, x, y, cw, ch) {
  const d = new Uint8Array(cw * ch * 3);
  for (let r = 0; r < ch; r++) d.set(im.d.subarray(((y + r) * im.w + x) * 3, ((y + r) * im.w + x + cw) * 3), r * cw * 3);
  return { w: cw, h: ch, d };
}

function mirror(im) {
  const d = new Uint8Array(im.d.length);
  for (let y = 0; y < im.h; y++) for (let x = 0; x < im.w; x++) {
    const s = (y * im.w + x) * 3, t = (y * im.w + (im.w - 1 - x)) * 3;
    d[t] = im.d[s]; d[t + 1] = im.d[s + 1]; d[t + 2] = im.d[s + 2];
  }
  return { w: im.w, h: im.h, d };
}

// --- сжатие как в Pillow (Resample.c, BILINEAR, 8 бит, фиксированная точка 22 бита) ---

const PREC = 22;

function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const tmp = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      tmp[x] = w; ww += w;
    }
    for (let x = 0; x < ksize; x++) {
      const k = x < xmax && ww !== 0 ? tmp[x] / ww : 0;
      kk[xx * ksize + x] = k < 0 ? Math.trunc(-0.5 + k * (1 << PREC)) : Math.trunc(0.5 + k * (1 << PREC));
    }
    bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

const clip8 = s => { const v = Math.floor(s / (1 << PREC)); return v < 0 ? 0 : v > 255 ? 255 : v; };

function resizeSquare(im, out) {
  const H = coeffs(im.w, out), V = coeffs(im.h, out);
  const yFirst = V.bounds[0];
  const yLast = V.bounds[out * 2 - 2] + V.bounds[out * 2 - 1];
  const th = yLast - yFirst;
  // горизонтальный проход только по нужным строкам
  const tmp = new Uint8Array(out * th * 3);
  for (let y = 0; y < th; y++) {
    const row = (y + yFirst) * im.w * 3;
    for (let xx = 0; xx < out; xx++) {
      const xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1], k0 = xx * H.ksize;
      let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
      for (let x = 0; x < xmax; x++) {
        const k = H.kk[k0 + x], p = row + (x + xmin) * 3;
        s0 += im.d[p] * k; s1 += im.d[p + 1] * k; s2 += im.d[p + 2] * k;
      }
      const o = (y * out + xx) * 3;
      tmp[o] = clip8(s0); tmp[o + 1] = clip8(s1); tmp[o + 2] = clip8(s2);
    }
  }
  const d = new Uint8Array(out * out * 3);
  for (let yy = 0; yy < out; yy++) {
    const ymin = V.bounds[yy * 2] - yFirst, ymax = V.bounds[yy * 2 + 1], k0 = yy * V.ksize;
    for (let xx = 0; xx < out; xx++) {
      let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
      for (let y = 0; y < ymax; y++) {
        const k = V.kk[k0 + y], p = ((y + ymin) * out + xx) * 3;
        s0 += tmp[p] * k; s1 += tmp[p + 1] * k; s2 += tmp[p + 2] * k;
      }
      const o = (yy * out + xx) * 3;
      d[o] = clip8(s0); d[o + 1] = clip8(s1); d[o + 2] = clip8(s2);
    }
  }
  return { w: out, h: out, d };
}

function centerSquare(im) {
  const s = Math.min(im.w, im.h);
  return crop(im, Math.floor((im.w - s) / 2), Math.floor((im.h - s) / 2), s, s);
}

function toTensor(im) {
  const n = im.w * im.h, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = (im.d[i * 3 + c] / 255) * 2 - 1;
  return new ort.Tensor('float32', t, [1, 3, im.h, im.w]);
}

// --- номер из бит ---

const checkChar = c => ABC[[...c].reduce((s, ch, i) => s + (i + 1) * ABC.indexOf(ch), 0) % ABC.length];

function certFromBits(bits) {
  const o = DataLayer_Decode(bits, eccengine, 'P');
  if (!o.valid || o.schema !== 'BCH_5') return null;
  let s = '';
  for (let i = 0; i < 56; i += 7) s += String.fromCharCode(parseInt(o.data_binary.slice(i, i + 7), 2));
  if ([...s].some(c => !ABC.includes(c)) || checkChar(s.slice(0, 7)) !== s[7]) return null;
  const c = s.slice(0, 7);
  return `${c.slice(0, 3)}-${c.slice(3, 5)}-${c.slice(5)}`;
}

async function findMark(session, im, onStep) {
  const total = 2 * CROPS.length;
  let step = 0;
  for (const flip of [false, true]) {
    const base = flip ? mirror(im) : im;
    for (const k of CROPS) {
      const cw = Math.floor(base.w * k), ch = Math.floor(base.h * k);
      const part = k === 1 ? base : crop(base, Math.floor((base.w - cw) / 2), Math.floor((base.h - ch) / 2), cw, ch);
      const x = toTensor(resizeSquare(centerSquare(part), SIZE));
      const out = await session.run({ image: x });
      const bits = Array.from(out.bits.data, v => v > 0);
      onStep(++step, total);
      const cert = certFromBits(bits);
      if (cert) return { cert, how: (flip ? 'картинка отзеркалена, ' : '') + (k === 1 ? 'целиком' : `по центральной части ${Math.round(k * 100)}%`) };
      await new Promise(r => setTimeout(r));
    }
  }
  return null;
}

// --- интерфейс ---

const $ = s => document.querySelector(s);
const results = $('#results');
const status = $('#status');
let queue = Promise.resolve();

function card(file) {
  const el = document.createElement('div');
  el.className = 'card';
  const img = document.createElement('img');
  img.alt = '';
  img.src = URL.createObjectURL(file);
  const body = document.createElement('div');
  body.className = 'body';
  body.innerHTML = '<div class="name"></div><div class="verdict wait">Жду очереди…</div><div class="note"></div>';
  body.querySelector('.name').textContent = file.name || 'картинка из буфера обмена';
  el.append(img, body);
  results.prepend(el);
  return {
    set(cls, text, note) {
      const v = body.querySelector('.verdict');
      v.className = 'verdict ' + cls;
      v.textContent = text;
      const n = body.querySelector('.note');
      n.textContent = '';
      if (note instanceof Node) n.append(note); else if (note) n.textContent = note;
    },
  };
}

function handle(files) {
  for (const file of files) {
    if (!file.type.startsWith('image/') && !/\.(png|jpe?g|webp|bmp|gif|avif)$/i.test(file.name || '')) continue;
    const c = card(file);
    queue = queue.then(() => check(file, c)).catch(() => {});
  }
}

async function check(file, c) {
  let session;
  try {
    c.set('wait', 'Загружаю модель проверки…');
    session = await loadModel(p => c.set('wait', `Загружаю модель проверки: ${Math.round(p * 100)}%. Это один раз, дальше быстро.`));
  } catch (e) {
    c.set('err', 'Не удалось загрузить модель проверки', 'Проверьте интернет и обновите страницу. ' + e.message);
    return;
  }
  let im;
  try { im = await decodeFile(file); } catch (e) {
    c.set('err', 'Не удалось открыть файл', 'Подходят PNG, JPG и WebP.');
    return;
  }
  const t0 = performance.now();
  const found = await findMark(session, im, (i, n) => c.set('wait', `Ищу метку: ${i} из ${n}`));
  if (found) {
    const note = document.createElement('span');
    const a = document.createElement('a');
    a.href = `https://www.copytrust.ru/item/view/${found.cert}`;
    a.target = '_blank'; a.rel = 'noopener';
    a.textContent = 'Открыть свидетельство на copytrust.ru';
    note.append(a, document.createTextNode(` · Метка прочитана: ${found.how}.`));
    c.set('ok', `Метка найдена. Свидетельство № ${found.cert}`, note);
  } else {
    c.set('none', 'Метки нет',
      'Метки нет в картинках, которые не размечали. Ещё её не прочитать, если картинку перерисовали, сфотографировали с экрана или сильно уменьшили внутри коллажа.');
  }
  console.log(file.name, found, Math.round(performance.now() - t0) + ' мс');
}

const drop = $('#drop');
const input = $('#file');
drop.addEventListener('click', () => input.click());
drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
input.addEventListener('change', () => { handle(input.files); input.value = ''; });
['dragenter', 'dragover'].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); if (ev === 'drop' || e.target === document.documentElement) drop.classList.remove('over'); }));
document.addEventListener('drop', e => {
  const files = [...(e.dataTransfer.files || [])];
  if (files.length) handle(files);
  else status.textContent = 'Перетащите сам файл картинки с компьютера. Картинку прямо из другой вкладки браузера сохраните или скопируйте и вставьте через Ctrl+V.';
});
document.addEventListener('paste', e => {
  const files = [...(e.clipboardData.items || [])].filter(i => i.kind === 'file').map(i => i.getAsFile()).filter(Boolean);
  if (files.length) handle(files);
});

// подгружаем модель заранее, пока человек ищет картинку
function preload() {
  status.textContent = 'Загружаю модель проверки…';
  loadModel(p => { status.textContent = `Загружаю модель проверки: ${Math.round(p * 100)}%. Это один раз, дальше быстро.`; })
    .then(() => { status.textContent = 'Модель проверки загружена, можно проверять.'; })
    .catch(() => { status.textContent = 'Модель проверки не загрузилась. Проверьте интернет: при следующей картинке попробую ещё раз.'; });
}
preload();

// для автотеста: window.__check(file) → {cert, how} | null
window.__check = async file => findMark(await loadModel(() => {}), await decodeFile(file), () => {});
