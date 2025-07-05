import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
try:
    model = load_model("quick_model.h5")  # ✅ your correct filename
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# Define class names (update if your model uses different ones)
CLASSES = ['Healthy', 'Coccidiosis', 'Salmonella', 'Newcastle Disease']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output', methods=['POST'])
def output():
    if 'file' not in request.files:
        return "❌ No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "❌ No file selected."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Load and preprocess image
        img = image.load_img(filepath, target_size=(64, 64))  # ✅ Match model input
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        if model:
            prediction = model.predict(img_array)
            predicted_class = CLASSES[np.argmax(prediction)]
        else:
            predicted_class = "Model not loaded."

    except Exception as e:
        return f"❌ Error during prediction: {e}"

    return render_template('index.html', prediction=predicted_class, image_file=filename)

if __name__ == '__main__':
    app.run(debug=True)
