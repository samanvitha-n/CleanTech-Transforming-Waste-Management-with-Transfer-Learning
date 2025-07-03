from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your VGG16 model
model = load_model('vgg16_model.h5')

# Class labels
classes = ['Biodegradable', 'Recyclable', 'Trash']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize like in your notebook

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return predicted_class, confidence

# ✅ Home page route (renders home.html)
@app.route('/')
def home():
    return render_template('home.html')

# ✅ Upload & Predict route (renders index.html for GET, predicts for POST)
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prediction, confidence = model_predict(file_path)

            return render_template('portfolio-details.html',
                                   prediction=prediction,
                                   confidence=confidence,
                                   user_image=file_path)

    # GET request – show upload form again (index.html)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
