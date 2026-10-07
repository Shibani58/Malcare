import os
from datetime import datetime

import numpy as np
import tensorflow.keras
from flask import Flask, redirect, render_template, request
from PIL import Image, ImageOps

from questions import main

app = Flask(__name__)

np.set_printoptions(suppress=True)
tb_model = tensorflow.keras.models.load_model('keras_model.h5')
malaria_model = tensorflow.keras.models.load_model('malaria_model.h5')

# Pillow 10 removed Image.ANTIALIAS; LANCZOS is the same filter.
RESAMPLE = getattr(Image, 'Resampling', Image).LANCZOS

# ---------------------------------------------------------------------------
# Patient records
# Firebase is optional and configured through environment variables, so no credentials live in the code.
# Without them, records are kept in memory for the current run only.
# ---------------------------------------------------------------------------
FIREBASE_CONFIG = {
    "apiKey": os.environ.get("FIREBASE_API_KEY"),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.environ.get("FIREBASE_DATABASE_URL"),
    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
}

db = None
if all(FIREBASE_CONFIG.values()):
    import pyrebase

    db = pyrebase.initialize_app(FIREBASE_CONFIG).database()

_local_records = []

FIELDS = ["date", "firstname", "lastname", "age", "height", "weight", "diagnosis", "prob"]


def add_patient(entry):
    record = dict(zip(FIELDS, entry))
    if db is not None:
        db.push(record)
    else:
        _local_records.append(record)


def get_patients():
    """Newest first, as lists in FIELDS order (the shape the templates expect)."""
    if db is not None:
        data = db.get().val() or {}
        records = list(data.values())
    else:
        records = _local_records
    return [[r[f] for f in FIELDS] for r in reversed(records)]


def get_numbers():
    patients = get_patients()
    uninfected = sum(1 for p in patients if p[6] == "Uninfected")
    return [uninfected, len(patients) - uninfected]


def get_age_data():
    """Infected patients per age band: <20, 20-39, 40-59, 60-79, 80+."""
    bands = [0, 0, 0, 0, 0]
    for p in get_patients():
        if p[6] in ("Tuberculosis", "Malaria"):
            bands[min(int(p[3]) // 20, 4)] += 1
    return bands


# ---------------------------------------------------------------------------
# Image classification
# ---------------------------------------------------------------------------
def predict(model, upload, grayscale):
    """Centre-crop to 224x224, scale to [-1, 1] and return the two class probabilities."""
    image = Image.open(upload.stream)
    if grayscale:
        # The TB model was trained on X-rays: one channel repeated three times.
        image = ImageOps.fit(image.convert("L"), (224, 224), RESAMPLE)
        array = np.stack((np.asarray(image),) * 3, axis=-1)
    else:
        image = ImageOps.fit(image.convert("RGB"), (224, 224), RESAMPLE)
        array = np.asarray(image)
    data = ((array.astype(np.float32) / 127.0) - 1)[np.newaxis, ...]
    return model.predict(data)[0]


def diagnose(model, positive_label, template, grayscale):
    upload = request.files.get('memory')
    if upload is None or upload.filename == "":
        return render_template(template, errorMessage="Please upload either a jpeg or png image.")
    try:
        result = predict(model, upload, grayscale)
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(template, errorMessage="Please upload either a jpeg or png image.")

    if result[0] > result[1]:
        diagnosis, prob = "Uninfected", round(float(result[0]) * 100, 2)
    else:
        diagnosis, prob = positive_label, round(float(result[1]) * 100, 2)

    form = request.form
    add_patient([datetime.now().strftime("%d/%m/%Y %H:%M:%S"), form.get("firstname", ""), form.get("lastname", ""),
                 form.get("age", "0"), form.get("height", ""), form.get("weight", ""), diagnosis, str(prob)])
    return render_template('results.html', diagnosis=diagnosis, prob=prob, result=result, text="")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    if request.method == 'POST':
        return diagnose(malaria_model, "Malaria", 'malaria.html', grayscale=False)
    return render_template('malaria.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        return diagnose(tb_model, "Tuberculosis", 'upload.html', grayscale=True)
    return render_template('upload.html')


@app.route('/trends')
def trends():
    k = get_numbers()
    k1 = get_age_data()
    return render_template('trends.html', a=k[0], b=k[1], p=k1[0], q=k1[1], r=k1[2], s=k1[3], t=k1[4])


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html', entries=get_patients())


@app.route('/download', methods=['GET', 'POST'])
def download():
    return redirect("/portfolio")


@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            return render_template('ai.html', errorMessage="Please type in a question.")
        return render_template('ai.html', response=main(question))
    return render_template('ai.html')


@app.route('/forum', methods=['GET', 'POST'])
def forum():
    return redirect("/ai")


@app.route('/stats')
def stats():
    return render_template('stats.html')


@app.route('/404')
def error():
    return render_template('404.html')


if __name__ == "__main__":
    app.run()
