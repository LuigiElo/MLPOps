import time
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

url = 'https://gcp-api-464642206755.europe-west1.run.app'
payload = {}

for i in range(1000):
    try:
        r = requests.get(url, params=payload)
        r.raise_for_status()  # Raise an error for bad status codes
        logging.info(f"Request {i+1} successful: {r.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request {i+1} failed: {e}")
    time.sleep(0.1)  # Add a small delay to avoid overwhelming the server
