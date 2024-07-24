
# import data file
import numpy as np
import pandas as pd
import torch as t
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import csv
from torchvision import transforms
from PIL import Image
import argparse
from flask import Flask, request, jsonify
from io import BytesIO
from flask_cors import CORS
import base64

import matplotlib.pyplot as plt


# Step 1: Initialize the weights
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # MNIST images are 28x28 pixels and there are 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.fc(x)
        return x

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


model = SimpleNN()

# Load the saved model
model.load_state_dict(t.load('C:\digit_classifer\simple_nn_model.pth'))
model.eval()  # Set the model to evaluation mode

#process data
transform = transforms.Compose([
    # transforms.Grayscale(),  # Ensure the image is in grayscale
    # transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    # transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to the same range as training data
])

@app.route('/predict', methods=['POST'])
# Function to preprocess and predict a digit from an image
def predict_digit():
    # image = Image.open(image_path)  # Load image
    # image = transform(image)  # Preprocess image
    print("Image received")
    data = request.get_json()
    image_data = data['image']
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
     # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    image = np.array(image)
    # Flatten to a 1D array of shape [784]
    image_flattened = image.flatten()
    image = t.tensor(image_flattened, dtype=t.long)

    # image = transform(image_data)  # Preprocess image
    image_normalized = (image / 255.0) * 2 - 1

    image = image_normalized.unsqueeze(0)  # Add batch dimension
    with t.no_grad():
        output = model(image)
        _, predicted = t.max(output.data, 1)
        # return predicted.item()  # Return the predicted digit
        return jsonify({'predictedDigit': predicted.item()})



# Example usage

# image_path = 'C:\digit_classifer\mnist_7.png'
# predicted_digit = predict_digit(image_path, model)
# print(f'The predicted digit is: {predicted_digit}')

if __name__ == '__main__':
    app.run(debug=True)
