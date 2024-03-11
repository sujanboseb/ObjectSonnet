import streamlit as st
import requests
import base64
import os

def process_image_link(image_link):
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

    return generate_output(image_path)

def query_detr(filename):
    API_URL_DETR = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
    API_KEY_DETR = "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"  # Replace with your actual API key for DETR
    headers_detr = {"Authorization": f"Bearer {API_KEY_DETR}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_DETR, headers=headers_detr, data=data)
    return response.json()

def generate_poem(labels):
    POEM_API_URL = "https://api-inference.huggingface.co/models/felixhusen/poem"
    POEM_API_KEY = "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"  # Replace with your actual API key for the poem generator
    POEM_HEADERS = {"Authorization": f"Bearer {POEM_API_KEY}"}

    # Combine labels into a single sentence
    sentence = ", ".join(labels)

    # Use the combined sentence as input for the poem generator
    poem_payload = {"inputs": sentence}
    poem_response = requests.post(POEM_API_URL, headers=POEM_HEADERS, json=poem_payload)

    # Process the poem_response.json() list
    poem_generated = " ".join(item.get('generated_text', 'Poem generation failed') for item in poem_response.json() if isinstance(item, dict))

    return {'labels': labels, 'poem_generated': poem_generated}

def generate_output(image_path):
    output_detr = query_detr(image_path)
    labels = [prediction.get("label", "Unknown") for prediction in output_detr]
    poem_output = generate_poem(labels)
    return poem_output

def main():
    st.title("Object Sonnet Generator")

    # User input: Paste the link from online
    image_link = st.text_input("Paste the link from online:", "")

    if st.button("Generate Poem"):
        # Process the image link and generate output
        output = process_image_link(image_link)

        # Display the output
        st.write("Labels:", output['labels'])
        st.write("Generated Poem:", output['poem_generated'])

if __name__ == '__main__':
    main()
