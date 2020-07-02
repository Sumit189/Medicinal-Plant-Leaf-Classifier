from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import os
import pandas as pd
import shutil
from PIL import Image, ImageOps
import numpy as np

tf.set_random_seed(0)
np.set_printoptions(suppress=True)
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = 'models/leafnet.h5'
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    tf.keras.backend.set_session(session)
    model = tf.keras.models.load_model(MODEL_PATH)
    model._make_predict_function()
    df2 = pd.read_csv("dataset_rand.csv")

print("Model loaded....")
with open('labels.txt', 'r') as f:
    labels = f.readlines()

@app.route('/')
def hello_world():
    shutil.rmtree('uploads/')
    os.mkdir('uploads')
    path = 'static/'
    names = ['minimize.css', 'plant.png', 'plant.css', 'style.css', 'favicon.ico', 'TestSet.zip']
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file != names[0] and file != names[1] and file != names[2] and file != names[3] and file != names[4] and file != names[5]:
                files.append(file)
    for f in files:
        os.remove('static/' + f)
    names_to_send= list(map(lambda s: s.strip(), labels))
    return render_template('index.html',labels=names_to_send)


@app.route('/', methods=['POST'])
def predictor():
    if request.method == 'POST':
        image = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, "uploads", secure_filename(image.filename)
        )
        file_path2 = os.path.join(
            basepath, "static", secure_filename(image.filename)
        )
        image.save(file_path)
        shutil.copyfile(file_path,file_path2)
        result=whatisthat(file_path)
        df=pd.read_csv('Plants_Medical_Record.csv')
        use=list(df[df['Species']==result]['MedicinalUse'])
        return render_template('plant.html', result=result, imagename=secure_filename(image.filename),muse=use[0])


def whatisthat(img_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    with session.graph.as_default():
        tf.keras.backend.set_session(session)
        prediction = model.predict(data)
    res_val=np.argmax(prediction)
    return labels[res_val].strip('\n')


if __name__ == '__main__':
    app.run(debug=False)
