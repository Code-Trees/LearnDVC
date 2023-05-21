from flask import Flask, render_template, request
import torch
from torchvision import transforms,models
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

app = Flask(__name__,template_folder='template')

# Load the PyTorch model
model = models.alexnet(pretrained=True)
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Load the image and apply the transformation pipeline
    img = Image.open(file)
    img = transform(img)
    img = img.unsqueeze(0)

    with open('imagenet_class.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Make a prediction using the PyTorch model and return the result
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        label = predicted.item()

    return render_template('index.html', prediction= str(classes[label]))

if __name__ == '__main__':
    app.run(debug=True)