/* MalCare browser demo: runs the original Keras models (converted to TensorFlow.js) and a hybrid-retrieval
 * question-answering engine (qa.js) entirely on the client. Nothing is uploaded or stored. */

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
let image = null; // HTMLImageElement ready for analysis

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
  // Start downloading the language models as soon as the visitor shows interest in asking.
  $('question').addEventListener('focus', () => getQa().then((q) => q.engine.prepare()).catch(() => {}), { once: true });
  document.querySelectorAll('.chip').forEach((c) => c.addEventListener('click', () => {
    $('question').value = c.textContent;
    askQuestion(c.textContent);
  }));
}

// ---------- Question answering (hybrid retrieval + neural reader, see qa.js) ----------

let qa = null;

/** Loads qa.js on first use so the image demo is not slowed down by the language models. */
async function getQa() {
  if (!qa) {
    const mod = await import(new URL('qa.js?v=3', document.baseURI).href);
    qa = { engine: new mod.QaEngine((msg) => { $('qa-status').textContent = msg; }), render: mod.renderAnswer };
  }
  return qa;
}

async function askQuestion(question) {
  const box = $('answer');
  box.hidden = false;
  if (!question.trim()) {
    box.innerHTML = '<p class="muted">Please type a question.</p>';
    return;
  }
  box.innerHTML = '<p class="muted">Thinking…</p>';
  try {
    const { engine, render } = await getQa();
    box.innerHTML = render(await engine.ask(question));
  } catch (err) {
    console.error(err);
    box.innerHTML = '<p class="muted">Something went wrong while answering. Please try again.</p>';
  }
}

// ---------- Start ----------
wireUi();
selectTest('malaria');
loadModels();
