import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/plant_disease_model.pth'
CLASS_NAMES_PATH = 'models/class_names.txt'
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class names
model = None
class_names = []
model_loaded = False
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    model = models.resnet50(weights=None)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = eval(f.read())
    if not class_names:
        raise ValueError("Class names list is empty")
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    model_loaded = True
    logger.info(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"Error loading model or class names: {e}")
    flash_message = "Model not loaded. Please run 'train_model.py' first."
    logger.warning(flash_message)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        logger.warning("No file part in request")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        logger.warning("No file selected")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
        except Exception as e:
            flash(f"Error saving file: {str(e)}")
            logger.error(f"Error saving file: {e}")
            return redirect(url_for('index'))

        if not model_loaded:
            flash("Model not loaded. Please run 'train_model.py' first.")
            logger.error("Prediction attempted but model not loaded")
            return redirect(url_for('index'))

        try:
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE, non_blocking=True)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_disease = class_names[predicted_class.item()]
                confidence = confidence.item() * 100

            result = {
                'disease': predicted_disease,
                'confidence': f"{confidence:.2f}%",
                'image_url': url_for('uploaded_file', filename=filename)
            }
            logger.info(f"Prediction: {predicted_disease} ({confidence:.2f}%)")

            return render_template('index.html', result=result, uploaded_image=filename)
        except Exception as e:
            flash(f"Prediction error: {str(e)}")
            logger.error(f"Prediction error: {e}")
            return redirect(url_for('index'))
    else:
        flash('Allowed image types: png, jpg, jpeg, gif')
        logger.warning(f"Invalid file type: {file.filename}")
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        flash(f"Error serving file: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Flask app failed to start: {e}")
        raise