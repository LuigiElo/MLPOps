import os
import requests
import streamlit as st
# from google.cloud import run_v2
from PIL import Image

BACKEND_URL = "https://gcp-api-464642206755.europe-west1.run.app"

#def get_backend_url():
    #Get the URL of the backend
#    parent = "project/My-First-Project/locations/europe-west1"
#    client = run_v2.ServicesClient()
#    services = client.list_services(parent=parent)
#    for service in services:
#        if service.name.split("/")[-1] == "production-model":
#            return service.uri
#    return os.environ.get("BACKEND", None)

def segment_image(image):
    """Send the image to the backend for segmentation"""
    predict_url = "{}/predict".format(BACKEND_URL)
    response = requests.post(predict_url, files={"image": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main():
    """Main function of the Streamlit frontend"""
    # backend = get_backend_url()
    backend = "https://gcp-api-464642206755.europe-west1.run.app" # Cloud URL
    if backend is None:
        raise ValueError("Backend service is not found")

    st.title("Image Segmentation")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        segmented_image = segment_image(image)

        if segmented_image is not None:
            # Uploaded image
            st.image(image, caption="Uploaded image", use_column_width=True)

            # Load the segmented image
            segmented_image = Image.open(io.BytesIO(segmented_image))

            # Segmented mask
            st.image(segmented_image, caption="Segmentation Mask", use_column_width=True)
        else:
            st.write("Failed to get segmentation result")

if __name__ == "__main__":
    main()
