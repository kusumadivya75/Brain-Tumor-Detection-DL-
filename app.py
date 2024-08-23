import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img_path):
    # Read the image using OpenCV
    image = cv2.imread(img_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image from OpenCV to PIL format
    image = Image.fromarray(image)
    # Resize the image to match the input size of the model
    image = image.resize((64, 64))
    # Convert the PIL Image to numpy array
    image = np.array(image)
    # Scale pixel values to [0, 1]
    image = image / 255.0
    # Add a batch dimension
    input_img = np.expand_dims(image, axis=0)
    # Predict the class
    predictions = model.predict(input_img)
    class_index = np.argmax(predictions, axis=-1)
    return class_index[0]  # Return the class index as a scalar

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        class_index = getResult(file_path)
        result = get_className(class_index)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)