import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import tifffile as tiff
from utils import open_image,postprocess_result,create_rgb_composite,load_multispectral_image
import cv2
# Initialize Flask application
app = Flask(__name__)

# Folder to store uploaded and segmented images
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load your deep learning model (adjust path as needed)
model = tf.keras.models.load_model('pretrained_model.keras')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and display results
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
        file.save(filepath)
        # Preprocess the image
        img = open_image(filepath)
        # Run the image through the model
        prediction = model.predict(img)

        # Postprocess the result and save the segmented image
        result_img = postprocess_result(prediction)
       
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], "output.png")
        multispectral_bands, _ = load_multispectral_image(filepath)
        rgb_img =create_rgb_composite(img, red=3, green=2, blue=1)
        # print(rgb_img.shape)
        image_name = filename.split(".")[0]+".png"
        
        rgb_path =os.path.join(os.path.join(app.config['RESULT_FOLDER'],"rgb"),image_name ) 
        os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
        rgb_img = np.squeeze(rgb_img) *255
        result = cv2.imwrite(rgb_path, rgb_img) 
        # print(result)
        result_img.save(result_filepath)

        # Render the template with the uploaded and segmented images
        return render_template('index.html', uploaded_image=rgb_path, segmented_image=result_filepath,output_image_name=image_name, input_image_name = filename)

if __name__ == '__main__':
    app.run(debug=True)
