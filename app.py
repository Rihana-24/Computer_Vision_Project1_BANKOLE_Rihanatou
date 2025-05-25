from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from werkzeug.utils import secure_filename

from models.cnn_pytorch import get_pretrained_model  # Make sure this module is present

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the models
model_path_tf = 'Rihanatou_BANKOLE_model.h5'
model_path_pth = 'Rihanatou_BANKOLE_model.torch'

# --- TensorFlow Model ---
if not os.path.exists(model_path_tf):
    raise FileNotFoundError(f"The file {model_path_tf} was not found.")
model_tf = tf.keras.models.load_model(model_path_tf)

# --- PyTorch Model ---
model_pth = get_pretrained_model()
if not os.path.exists(model_path_pth):
    raise FileNotFoundError(f"The file {model_path_pth} was not found.")
model_pth.load_state_dict(torch.load(model_path_pth, map_location=torch.device('cpu')))
model_pth.eval()

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Preprocessing for TensorFlow
def preprocess_tf(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Preprocessing for PyTorch
def preprocess_pt(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files or 'framework' not in request.form:
            return render_template('index.html', prediction='Missing file or framework.', image_url=None)
        
        file = request.files['file']
        framework = request.form['framework']

        if file.filename == '':
            return render_template('index.html', prediction='No file selected.', image_url=None)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image_url = '/' + filepath

        if framework == 'tensorflow':
            img = preprocess_tf(filepath)
            preds = model_tf.predict(img)
            pred_class = class_names[np.argmax(preds[0])]
        elif framework == 'pytorch':
            img = preprocess_pt(filepath)
            with torch.no_grad():
                outputs = model_pth(img)
                _, predicted = torch.max(outputs, 1)
                pred_class = class_names[predicted.item()]
        else:
            pred_class = 'Unsupported framework.'

        prediction = f'Predicted class: {pred_class}'

    return render_template('index.html', prediction=prediction, image_url=image_url)

# Start the server
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
