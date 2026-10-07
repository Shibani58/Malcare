# MalCare – AI-Assisted Malaria & Tuberculosis Screening

MalCare classifies **blood-smear cell images for malaria** and **chest X-rays for tuberculosis** with two
transfer-learning neural networks, and answers questions about both diseases with a **retrieve-then-read**
question-answering pipeline (hybrid BM25 + MiniLM retrieval and a DistilBERT reader).

**🌐 Live demo: https://shibani58.github.io/Malcare/**: runs entirely in your browser with TensorFlow.js.
No images or personal data leave your device.

> ⚠️ Research and learning project. The models are not clinically validated and must not be used for diagnosis.

---

## Features

| Feature | How it works |
|---|---|
| **Malaria screening** | Blood-smear cell image → *Malaria* / *Uninfected* with a confidence score |
| **Tuberculosis screening** | Chest X-ray → *Tuberculosis* / *Uninfected* with a confidence score |
| **Question answering** | Hybrid retrieval (BM25 + MiniLM embeddings, fused with Reciprocal Rank Fusion) and a DistilBERT reader that extracts the answer span |
| **Patient records & trends** *(Flask app)* | Stores each result with patient details and shows infection counts by age group |

## Models

Both classifiers use the same transfer-learning architecture: a **MobileNet** feature extractor →
global average pooling → a 100-unit dense layer → a 2-class softmax. They were built with Google Teachable
Machine and exported as Keras models (`malaria_model.h5`, `keras_model.h5`).

Pre-processing (identical in the Flask app and the browser demo): centre-crop to a square, resize to 224×224,
scale pixels to [-1, 1]. X-rays are converted to greyscale and repeated across the three channels.

## Question answering (browser demo)

A retrieve-then-read pipeline built with [Transformers.js](https://huggingface.co/docs/transformers.js):

1. **Passages:** the malaria and tuberculosis articles are split into overlapping windows of up to three sentences.
2. **Hybrid retrieval:** every passage is ranked by **BM25** (keyword relevance) and by cosine similarity of
   **all-MiniLM-L6-v2** sentence embeddings (meaning), and the two rankings are merged with **Reciprocal Rank
   Fusion**. Paraphrased questions such as "How do people catch TB?" still find the right passage.
3. **Neural reader:** **DistilBERT fine-tuned on SQuAD** reads the top 3 passages and extracts the answer span with a
   confidence score. Because the reader is extractive, it can only quote its sources and cannot invent facts.
4. **Fallback:** if the models cannot be downloaded, answers use BM25 alone.

The models (about 90 MB, quantised) download when a visitor first starts asking a question and are cached by the
browser. The Flask app keeps its original TF-IDF engine (`questions.py`).

## Two ways to run it

### 1. Browser demo (`web/`)
A static site deployed to GitHub Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml), which
converts the Keras models to TensorFlow.js with `tensorflowjs_converter`. Inference and question answering run on
the client; results are kept only for the current browser tab.

### 2. Flask web app
```bash
git clone https://github.com/Shibani58/Malcare.git
cd Malcare
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                         # open http://127.0.0.1:5000
```
Requires Python 3.9–3.11 (TensorFlow < 2.16, because the models are Keras 2 files). NLTK data is downloaded
automatically on first use.

**Patient records (optional).** To store results in Firebase Realtime Database, set `FIREBASE_API_KEY`,
`FIREBASE_AUTH_DOMAIN`, `FIREBASE_DATABASE_URL` and `FIREBASE_STORAGE_BUCKET`. Without them, records are kept in
memory for the current run only. Never commit credentials, and protect the database with authentication rules,
since it holds health data.

## Project structure

```
app.py               Flask routes, image pre-processing, patient records
questions.py         TF-IDF question answering (NLTK)
corpus/              Text documents used by the Q&A engine
malaria_model.h5     Malaria classifier (Keras)
keras_model.h5       Tuberculosis classifier (Keras)
templates/, static/  Flask web interface
web/                 Browser demo (TensorFlow.js)
```

## Tech stack
Python · TensorFlow / Keras · TensorFlow.js · Transformers.js (MiniLM, DistilBERT) · BM25 · Flask · NLTK · NumPy ·
Pillow · Firebase (optional) · GitHub Actions

## Limitations
- The models have not been evaluated on a held-out clinical test set, so their accuracy is not reported here.
- The malaria model expects a single segmented red-blood-cell image, not a full microscope slide.
- The TB model expects a frontal chest X-ray.

## Credits
- The Flask app's TF-IDF question-answering engine follows the design of Harvard's CS50 AI "Questions" project.
- Browser Q&A models: `Xenova/all-MiniLM-L6-v2` and `Xenova/distilbert-base-cased-distilled-squad` on Hugging Face.
- Q&A documents are from Wikipedia (CC BY-SA).
