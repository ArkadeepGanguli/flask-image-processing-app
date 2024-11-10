from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to save original images
app.config['FILTERED_FOLDER'] = 'static/filtered'  # Folder to save filtered images

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FILTERED_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    filtered_image = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        # Save the original image
        filename = secure_filename(file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)

        # Apply the selected filter
        filter_name = request.form.get('filter')
        image = cv2.imread(original_path)

        # Apply selected filter
        if filter_name == 'grayscale':
            filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif filter_name == 'blur':
            filtered_image = cv2.GaussianBlur(image, (15, 15), 0)
        elif filter_name == 'edge_detection':
            filtered_image = cv2.Canny(image, 100, 200)
        elif filter_name == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            filtered_image = cv2.filter2D(image, -1, kernel)
        elif filter_name == 'invert':
            filtered_image = cv2.bitwise_not(image)
        elif filter_name == 'sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            filtered_image = cv2.transform(image, sepia_filter)
            filtered_image = np.clip(filtered_image, 0, 255)  # Ensure pixel values are in range
        elif filter_name == 'oil_painting':
            filtered_image = cv2.xphoto.oilPainting(image, 7, 1).astype(np.uint8)
        elif filter_name == 'cartoon':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(image, 9, 300, 300)
            filtered_image = cv2.bitwise_and(color, color, mask=edges).astype(np.uint8)
        elif filter_name == 'sketch':
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inverted_image = cv2.bitwise_not(gray_image)
            sketch = cv2.divide(gray_image, inverted_image, scale=256.0)
            filtered_image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        elif filter_name == 'emboss':
            kernel = np.array([[-2, -1, 0],
                            [-1,  1, 1],
                            [0,  1, 2]])
            filtered_image = cv2.filter2D(image, -1, kernel).astype(np.uint8)

        # Save the filtered image
        filtered_filename = f"filtered_{filename}"
        filtered_path = os.path.join(app.config['FILTERED_FOLDER'], filtered_filename)

        # Handle grayscale and colored images separately for saving
        if filter_name in ['grayscale', 'edge_detection']:
            cv2.imwrite(filtered_path, filtered_image)
        else:
            cv2.imwrite(filtered_path, cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))

        # Pass paths to display in the template
        original_image = f"uploads/{filename}"
        filtered_image = f"filtered/{filtered_filename}"

    return render_template('index.html', original_image=original_image, filtered_image=filtered_image)

if __name__ == '__main__':
    app.run(debug=True)
