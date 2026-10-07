/* MalCare browser demo: runs the original Keras models (converted to TensorFlow.js) and a TF-IDF
 * question-answering engine entirely on the client. Nothing is uploaded or stored. */

const TESTS = {
  malaria: {
    name: 'Malaria',
    model: 'models/malaria/model.json',
    hint: 'A single red-blood-cell image (PNG or JPG)',
    sample: 'samples/malaria-cell.png',
  },
  tb: {
    name: 'Tuberculosis',
    model: 'models/tb/model.json',
    hint: 'A chest X-ray (PNG or JPG)',
    sample: null,
  },
};

const SIZE = 224;
const models = {};
const results = [];
let current = 'malaria';
let image = null; // HTMLImageElement or ImageBitmap ready for analysis

const $ = (id) => document.getElementById(id);

// ---------- Models ----------

async function loadModels() {
  const status = $('model-status');
  try {
    await tf.ready();
    const [malaria, tb] = await Promise.all([tf.loadLayersModel(TESTS.malaria.model), tf.loadLayersModel(TESTS.tb.model)]);
    models.malaria = malaria;
    models.tb = tb;
    // Warm up so the first real prediction is fast.
    tf.tidy(() => { malaria.predict(tf.zeros([1, SIZE, SIZE, 3])); tb.predict(tf.zeros([1, SIZE, SIZE, 3])); });
    status.textContent = `Models ready · running on ${tf.getBackend().toUpperCase()} in your browser`;
    status.className = 'model-status ok';
  } catch (err) {
    console.error(err);
    status.textContent = 'The models could not be loaded. Please refresh the page.';
    status.className = 'model-status err';
  }
  updateButtons();
}

/** Same preprocessing as the Flask app: centre-crop to a square (PIL ImageOps.fit), 224x224, scale to [-1, 1]. */
function toInput(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const side = Math.min(w, h);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, SIZE, SIZE);
  // Greyscale X-rays become R=G=B, matching the Flask app's channel stacking.
  return tf.tidy(() => tf.browser.fromPixels(canvas, 3).toFloat().div(127).sub(1).expandDims(0));
}

async function analyse() {
  if (!image || !models[current]) return;
  $('analyse').disabled = true;
  const input = toInput(image);
  const output = models[current].predict(input);
  const [uninfected, infected] = await output.data();
  input.dispose();
  output.dispose();

  const test = TESTS[current];
  const positive = infected > uninfected;
  const label = positive ? test.name : 'Uninfected';
  const confidence = Math.max(infected, uninfected) * 100;
  renderResult(test, label, positive, confidence, uninfected, infected);
  results.unshift({ time: new Date(), test: test.name, label, positive, confidence });
  renderHistory();
  updateButtons();
}

// ---------- UI ----------

function renderResult(test, label, positive, confidence, uninfected, infected) {
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  $('result').innerHTML = `
    <span class="muted small">${test.name} screening result</span>
    <div class="verdict ${positive ? 'pos' : 'neg'}">${label}</div>
    <div class="muted">Model confidence: <strong>${confidence.toFixed(1)}%</strong></div>
    <div class="bar-row"><span>Uninfected</span><div class="bar"><div style="width:${pct(uninfected)}"></div></div><span class="num">${pct(uninfected)}</span></div>
    <div class="bar-row"><span>${test.name}</span><div class="bar pos"><div style="width:${pct(infected)}"></div></div><span class="num">${pct(infected)}</span></div>
    <p class="small muted">A screening aid only. Confirm any result with a clinician and a laboratory test.</p>`;
}

function renderHistory() {
  const rows = results.map((h) => `
    <tr><td>${h.time.toLocaleTimeString()}</td><td>${h.test}</td>
    <td><span class="tag ${h.positive ? 'pos' : 'neg'}">${h.label}</span></td>
    <td class="num">${h.confidence.toFixed(1)}%</td></tr>`);
  $('history').innerHTML = rows.join('') || '<tr><td colspan="4" class="muted">No images analysed yet.</td></tr>';
  const count = (fn) => results.filter(fn).length;
  $('stats').innerHTML = results.length ? `
    <span class="stat"><strong>${results.length}</strong>analysed</span>
    <span class="stat"><strong>${count((h) => h.positive && h.test === 'Malaria')}</strong>malaria positive</span>
    <span class="stat"><strong>${count((h) => h.positive && h.test === 'Tuberculosis')}</strong>TB positive</span>
    <span class="stat"><strong>${count((h) => !h.positive)}</strong>uninfected</span>` : '';
}

function updateButtons() {
  $('analyse').disabled = !(image && models[current]);
  $('sample').hidden = !TESTS[current].sample;
}

function showImage(src) {
  const img = new Image();
  img.onload = () => {
    image = img;
    const preview = $('preview');
    preview.src = src;
    preview.hidden = false;
    $('drop-text').hidden = true;
    $('result').innerHTML = '<p class="muted">Press <strong>Analyse</strong> to classify this image.</p>';
    updateButtons();
  };
  img.onerror = () => {
    $('result').innerHTML = '<p class="muted">That file could not be read as an image. Please use a PNG or JPG.</p>';
  };
  img.src = src;
}

function selectTest(test) {
  current = test;
  document.querySelectorAll('.tab').forEach((t) => {
    const on = t.dataset.test === test;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', on);
  });
  $('drop-hint').textContent = TESTS[test].hint;
  image = null;
  $('preview').hidden = true;
  $('drop-text').hidden = false;
  $('file').value = '';
  $('result').innerHTML = '<p class="muted">The result will appear here.</p>';
  updateButtons();
}

function wireUi() {
  document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => selectTest(t.dataset.test)));
  $('file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) showImage(URL.createObjectURL(file));
  });
  const drop = $('drop');
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    const file = e.dataTransfer.files[0];
    if (file) showImage(URL.createObjectURL(file));
  });
  $('sample').addEventListener('click', () => TESTS[current].sample && showImage(TESTS[current].sample));
  $('analyse').addEventListener('click', analyse);
  $('ask-form').addEventListener('submit', (e) => { e.preventDefault(); askQuestion($('question').value); });
  document.querySelectorAll('.chip').forEach((c) => c.addEventListener('click', () => {
    $('question').value = c.textContent;
    askQuestion(c.textContent);
  }));
}

// ---------- Question answering (port of questions.py) ----------

// NLTK English stop-word list, as used by the original Python implementation.
const STOPWORDS = new Set(("i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him his " +
  "himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom this that that'll these " +
  "those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by " +
  "for with about against between into through during before after above below to from up down in out on off over under again further " +
  "then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t " +
  "can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't " +
  "hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn wasn't weren " +
  "weren't won won't wouldn wouldn't").split(' '));

const CORPUS_FILES = ['malaria', 'tuberculosis', 'artificial_intelligence', 'machine_learning', 'natural_language_processing',
  'neural_network', 'probability', 'python'];
const ANSWER_FILES = new Set(['malaria', 'tuberculosis']); // the demo only answers from the medical documents
let corpus = null;

function tokenize(text) {
  const words = text.toLowerCase().match(/[a-z0-9]+(?:['-][a-z0-9]+)*/g) || [];
  return words.filter((w) => !STOPWORDS.has(w));
}

function computeIdfs(docs) {
  const counts = new Map();
  for (const words of docs.values()) {
    for (const w of new Set(words)) counts.set(w, (counts.get(w) || 0) + 1);
  }
  const idfs = new Map();
  for (const [w, c] of counts) idfs.set(w, Math.log(docs.size / c));
  return idfs;
}

async function loadCorpus() {
  if (corpus) return corpus;
  const texts = await Promise.all(CORPUS_FILES.map((f) => fetch(`corpus/${f}.txt`).then((r) => r.text())));
  // Drop Wikipedia citation markers such as [1] or [citation needed].
  const files = new Map(CORPUS_FILES.map((f, i) => [f, texts[i].replace(/\[[^\]]{1,40}\]/g, '')]));
  const fileWords = new Map([...files].map(([f, t]) => [f, tokenize(t)]));
  corpus = { files, fileWords, fileIdfs: computeIdfs(fileWords) };
  return corpus;
}

async function askQuestion(question) {
  const box = $('answer');
  const query = new Set(tokenize(question));
  box.hidden = false;
  if (!query.size) {
    box.innerHTML = '<p class="muted">Please type a question with a few key words.</p>';
    return;
  }
  box.innerHTML = '<p class="muted">Searching…</p>';
  const { files, fileWords, fileIdfs } = await loadCorpus();

  // 1. Best document by TF-IDF.
  let best = null;
  let bestScore = -1;
  for (const [f, words] of fileWords) {
    if (!ANSWER_FILES.has(f)) continue;
    let score = 0;
    for (const q of query) {
      const tf = words.filter((w) => w === q).length;
      if (tf) score += tf * fileIdfs.get(q);
    }
    if (score > bestScore) { best = f; bestScore = score; }
  }

  // 2. Best sentences in that document by summed IDF, ties broken by query-term density.
  const sentences = new Map();
  for (const passage of files.get(best).split('\n')) {
    for (const s of passage.split(/(?<=[.!?])\s+(?=[A-Z0-9"“(])/)) {
      const tokens = tokenize(s);
      if (tokens.length && s.trim().length > 25) sentences.set(s.trim(), tokens);
    }
  }
  const idfs = computeIdfs(sentences);
  const ranked = [...sentences].map(([s, tokens]) => {
    let score = 0;
    let density = 0;
    for (const q of query) {
      if (tokens.includes(q)) {
        score += idfs.get(q);
        density += tokens.filter((t) => t === q).length / tokens.length;
      }
    }
    return { s, score, density };
  }).sort((a, b) => b.score - a.score || b.density - a.density);

  if (!ranked.length || ranked[0].score === 0) {
    box.innerHTML = '<p class="muted">I could not find an answer to that. Try asking about symptoms, causes, diagnosis or treatment.</p>';
    return;
  }
  const title = best === 'tuberculosis' ? 'Tuberculosis' : 'Malaria';
  const escape = (t) => t.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  box.innerHTML = `
    <span class="muted small">Best match · ${title} (Wikipedia)</span>
    <blockquote>${escape(ranked[0].s)}</blockquote>
    <span class="muted small">Also relevant</span>
    <ol>${ranked.slice(1, 3).map((r) => `<li>${escape(r.s)}</li>`).join('')}</ol>`;
}

// ---------- Start ----------
wireUi();
selectTest('malaria');
loadModels();
