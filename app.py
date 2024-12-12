from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io

# Initialize Flask App
app = Flask(__name__)

# Setup for storing uploaded files
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the UNet model (we're assuming the model is trained and saved as 'unet_model.pth')
# The model is defined earlier; here we just load it
model = torch.load('unet_model.pth')  # Assuming the model is saved as 'unet_model.pth'


# Helper function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and model processing
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    if image and allowed_file(image.filename):
        # Secure filename and save the image
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Process the image using the selected model
        model_choice = int(request.form['model_choice'])  # Retrieve the model choice (1 = UNet, etc.)
        
        if model_choice == 1:  # UNet model
            processed_image_path = process_with_unet(image_path)

        # Return the processed image path for display
        return jsonify({'processed_image': processed_image_path})

    return jsonify({'error': 'Invalid file'})

# Function to process the image with UNet
def process_with_unet(image_path):
    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Transform image (Resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Pass through the model
    with torch.no_grad():
        output = model(image_tensor)  # Get model output
        output = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

    # Convert output to an image and save
    output_image = np.transpose(output, (1, 2, 0))  # Convert to HWC format
    output_image = (output_image * 255).astype(np.uint8)  # Rescale to [0, 255]
    
    processed_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    Image.fromarray(output_image).save(processed_image_path)

    return processed_image_path

if __name__ == '__main__':
    # Create folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    app.run(debug=True)
