/* MalCare question answering: hybrid retrieval + neural reader, entirely in the browser.
 *
 *  1. The malaria and tuberculosis articles are split into short overlapping passages.
 *  2. Retrieval combines BM25 (lexical) with MiniLM sentence embeddings (semantic) using
 *     Reciprocal Rank Fusion, so paraphrased questions still find the right passage.
 *  3. A DistilBERT model fine-tuned on SQuAD reads the top passages and extracts the answer span.
 *     The reader can only quote the source text, so it cannot invent medical facts.
 *  If the neural models cannot be downloaded, answers fall back to BM25 alone.
 */

const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const EMBEDDER = 'Xenova/all-MiniLM-L6-v2';
const READER = 'Xenova/distilbert-base-cased-distilled-squad';
const DOCS = { malaria: 'Malaria', tuberculosis: 'Tuberculosis' };
const TOP_K = 3;
const RRF_K = 60;
// Reader scores from different passages are not directly comparable (each is normalised within its own passage),
// so the final choice also trusts the retrieval ranking: score x prior for fused ranks 1, 2 and 3.
const RANK_PRIOR = [1, 0.7, 0.4];

const STOPWORDS = new Set(('a an the and or but if of at by for with about against between into through during before after above ' +
  'below to from up down in out on off over under again then once here there when where why how all any both each few more most ' +
  'other some such no nor not only own same so than too very can will just should now is are was were be been being have has had ' +
  'having do does did doing i me my we our you your he him his she her it its they them their what which who whom this that these ' +
  'those am').split(' '));

const tokenize = (text) => (text.toLowerCase().match(/[a-z0-9]+(?:['-][a-z0-9]+)*/g) || []).filter((w) => !STOPWORDS.has(w));
const escapeHtml = (t) => t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------- Passages ----------

function buildPassages(doc, text) {
  const passages = [];
  const clean = text.replace(/\[[^\]]{1,40}\]/g, ''); // Wikipedia citation markers
  for (const paragraph of clean.split('\n')) {
    if (paragraph.trim().length < 80) continue; // headings and stubs
    const sentences = paragraph.trim().split(/(?<=[.!?])\s+(?=[A-Z0-9"“(])/);
    // Windows of up to three sentences, overlapping by one, so an answer is never cut off at a boundary.
    for (let i = 0; i < sentences.length; i += 2) {
      const passage = sentences.slice(i, i + 3).join(' ');
      if (passage.length > 60) passages.push({ doc, text: passage, tokens: tokenize(passage) });
      if (i + 3 >= sentences.length) break;
    }
  }
  return passages;
}

/** The reader's tokenizer re-spaces punctuation ("antigen - based"); find the exact span in the passage instead. */
function originalSpan(answer, text) {
  const pattern = answer.trim().split(/\s+/).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s*');
  const match = pattern && new RegExp(pattern).exec(text);
  return match ? match[0] : answer;
}

// ---------- BM25 ----------

function buildBm25(passages, k1 = 1.5, b = 0.75) {
  const df = new Map();
  for (const p of passages) for (const t of new Set(p.tokens)) df.set(t, (df.get(t) || 0) + 1);
  const n = passages.length;
  const avgLen = passages.reduce((s, p) => s + p.tokens.length, 0) / n;
  const idf = (t) => Math.log(1 + (n - (df.get(t) || 0) + 0.5) / ((df.get(t) || 0) + 0.5));
  return (query) => {
    const terms = [...new Set(tokenize(query))];
    return passages.map((p) => {
      let score = 0;
      for (const t of terms) {
        const tf = p.tokens.filter((w) => w === t).length;
        if (tf) score += idf(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * p.tokens.length) / avgLen));
      }
      return score;
    });
  };
}

// ---------- Engine ----------

export class QaEngine {
  constructor(onStatus) {
    this.onStatus = onStatus;
    this.ready = null;
  }

  /** Loads corpus, models and passage embeddings once; safe to call repeatedly. */
  prepare() {
    this.ready ??= this.#load();
    return this.ready;
  }

  async #load() {
    const texts = await Promise.all(Object.keys(DOCS).map((d) => fetch(`corpus/${d}.txt`).then((r) => r.text())));
    this.passages = Object.keys(DOCS).flatMap((d, i) => buildPassages(d, texts[i]));
    this.bm25 = buildBm25(this.passages);

    try {
      const { pipeline } = await import(TRANSFORMERS);
      const progress = new Map();
      const onProgress = (p) => {
        if (p.status === 'progress' && p.total) {
          progress.set(p.file + p.name, [p.loaded, p.total]);
          const [done, total] = [...progress.values()].reduce((a, [l, t]) => [a[0] + l, a[1] + t], [0, 0]);
          this.onStatus(`Downloading language models (first visit only)… ${Math.round((100 * done) / total)}%`);
        }
      };
      this.onStatus('Downloading language models (first visit only)…');
      [this.embedder, this.reader] = await Promise.all([
        pipeline('feature-extraction', EMBEDDER, { dtype: 'q8', progress_callback: onProgress }),
        pipeline('question-answering', READER, { dtype: 'q8', progress_callback: onProgress }),
      ]);

      this.vectors = [];
      for (let i = 0; i < this.passages.length; i += 16) {
        this.onStatus(`Indexing passages for semantic search… ${i}/${this.passages.length}`);
        const batch = this.passages.slice(i, i + 16).map((p) => p.text);
        const out = await this.embedder(batch, { pooling: 'mean', normalize: true });
        this.vectors.push(...out.tolist());
      }
      this.mode = 'neural';
      this.onStatus(`Ready · hybrid BM25 + MiniLM retrieval over ${this.passages.length} passages, DistilBERT reader`);
    } catch (err) {
      console.warn('Neural models unavailable, using BM25 only', err);
      this.mode = 'bm25';
      this.onStatus('Neural models could not be loaded, so answers use BM25 keyword retrieval only.');
    }
  }

  async ask(question) {
    await this.prepare();
    const bm25 = this.bm25(question);
    const order = (scores) => scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
    const bm25Rank = new Map(order(bm25).map((idx, r) => [idx, r + 1]));

    if (this.mode !== 'neural') {
      const best = order(bm25).slice(0, TOP_K).filter((i) => bm25[i] > 0);
      return { mode: 'bm25', passages: best.map((i) => ({ ...this.passages[i], bm25Rank: bm25Rank.get(i) })) };
    }

    const [q] = (await this.embedder([question], { pooling: 'mean', normalize: true })).tolist();
    const dense = this.vectors.map((v) => v.reduce((s, x, j) => s + x * q[j], 0));
    const denseRank = new Map(order(dense).map((idx, r) => [idx, r + 1]));

    // Reciprocal Rank Fusion: robust to the very different score scales of BM25 and cosine similarity.
    const fused = this.passages.map((_, i) => 1 / (RRF_K + bm25Rank.get(i)) + 1 / (RRF_K + denseRank.get(i)));
    const top = order(fused).slice(0, TOP_K);

    const candidates = [];
    for (const [rank, i] of top.entries()) {
      const { text } = this.passages[i];
      const { answer, score } = await this.reader(question, text);
      candidates.push({
        ...this.passages[i], answer: originalSpan(answer, text), score, weighted: score * RANK_PRIOR[rank],
        bm25Rank: bm25Rank.get(i), denseRank: denseRank.get(i),
      });
    }
    const best = candidates.reduce((a, c) => (c.weighted > a.weighted ? c : a));
    return { mode: 'neural', best, passages: candidates };
  }
}

// ---------- Rendering ----------

export function renderAnswer(result) {
  if (!result.passages.length) {
    return '<p class="muted">I could not find anything about that. Try asking about symptoms, causes, diagnosis or treatment.</p>';
  }
  const source = (p) => `${DOCS[p.doc]} (Wikipedia)`;
  const ranks = (p) => p.denseRank ? `BM25 #${p.bm25Rank} · semantic #${p.denseRank}` : `BM25 #${p.bm25Rank}`;

  if (result.mode !== 'neural') {
    const [first, ...rest] = result.passages;
    return `
      <span class="muted small">Most relevant passage · ${source(first)} · keyword search</span>
      <blockquote>${escapeHtml(first.text)}</blockquote>
      ${rest.length ? `<span class="muted small">Also relevant</span><ol>${rest.map((p) => `<li>${escapeHtml(p.text)}</li>`).join('')}</ol>` : ''}`;
  }

  const { best } = result;
  const confident = best.score >= 0.1;
  const highlighted = (() => {
    const at = best.text.indexOf(best.answer);
    if (!best.answer || at < 0) return escapeHtml(best.text);
    return escapeHtml(best.text.slice(0, at)) + `<mark>${escapeHtml(best.answer)}</mark>` + escapeHtml(best.text.slice(at + best.answer.length));
  })();
  const others = result.passages.filter((p) => p !== best);
  return `
    <span class="muted small">${confident ? 'Answer' : 'Low-confidence answer, please read the passage'} · ${source(best)}</span>
    <p class="answer-span">${escapeHtml(best.answer)}</p>
    <div class="confidence"><div class="bar"><div style="width:${(best.score * 100).toFixed(0)}%"></div></div>
      <span class="small muted">reader confidence ${(best.score * 100).toFixed(0)}%</span></div>
    <blockquote>${highlighted}</blockquote>
    <span class="muted small">Retrieved passages</span>
    <ol class="retrieved">
      <li><span class="rank">${ranks(best)}</span> <em>the passage above</em></li>
      ${others.map((p) => `<li><span class="rank">${ranks(p)}</span> ${escapeHtml(p.text)}</li>`).join('')}
    </ol>`;
}
