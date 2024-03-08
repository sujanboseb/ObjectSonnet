from flask import Flask, render_template, request, jsonify
import cv2
import os
import requests
import base64

app = Flask(__name__)

# Set up the camera
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image_link', methods=['POST'])
def process_image_link():
    # Get image link from the form
    image_link = request.form['image_link']

    # Check if the image link is in base64 format
    if image_link.startswith('data:image'):
        # Extract base64-encoded image data
        _, encoded_data = image_link.split(',')
        decoded_data = base64.b64decode(encoded_data)

        # Save the decoded image
        image_path = 'uploads/image_from_link.jpg'
        with open(image_path, 'wb') as f:
            f.write(decoded_data)
    else:
        # If it's a regular URL, download the image
        response = requests.get(image_link)
        image_path = 'uploads/image_from_link.jpg'
        with open(image_path, 'wb') as f:
            f.write(response.content)
    
    output = generate_output(image_path)
    
    # Return a JSON response with both labels and poem_generated
    return render_template('result.html', response=output)

def query_detr(filename):
    API_URL_DETR = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
    API_KEY_DETR = "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"  # Replace with your actual API key for DETR
    headers_detr = {"Authorization": f"Bearer {API_KEY_DETR}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_DETR, headers=headers_detr, data=data)
    return response.json()

# ...

def generate_poem(labels):
    # Combine labels into a single sentence
    sentence = ", ".join(labels)

    # Use the combined sentence as input for the poem generator
    POEM_API_URL = "https://api-inference.huggingface.co/models/felixhusen/poem"
    POEM_API_KEY = "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"  # Replace with your actual API key for the poem generator
    POEM_HEADERS = {"Authorization": f"Bearer {POEM_API_KEY}"}

    poem_payload = {"inputs": sentence}
    poem_response = requests.post(POEM_API_URL, headers=POEM_HEADERS, json=poem_payload)

    # Process the poem_response.json() list
    poem_generated = " ".join(item.get('generated_text', 'Poem generation failed') for item in poem_response.json() if isinstance(item, dict))

    return {'labels': labels, 'poem_generated': poem_generated}

# ...

def generate_output(image_path):
    output_detr = query_detr(image_path)
    labels = [prediction.get("label", "Unknown") for prediction in output_detr]
    poem_output = generate_poem(labels)
    return poem_output

if __name__ == '__main__':
    app.run(debug=True)
