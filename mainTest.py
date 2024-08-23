import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Read the image using OpenCV
image = cv2.imread('D:\\BRAIN TUMOR CLASSIFICATION\\pred\\pred0.jpg')

# Convert the image from BGR to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the OpenCV image (numpy array) to a PIL Image
img = Image.fromarray(image)

# Resize the image to match the input size of the model
img = img.resize((64, 64))

# Convert the PIL Image back to a numpy array
img = np.array(img)

# Normalize pixel values if your model expects data to be normalized
img = img / 255.0

# Add a batch dimension to the array
input_img = np.expand_dims(img, axis=0)

# Predict the class probabilities or logits
predictions = model.predict(input_img)

# Get the index of the highest probability class
result = np.argmax(predictions, axis=-1)

# Print the result
print(result)