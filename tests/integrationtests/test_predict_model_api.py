import os
from fastapi.testclient import TestClient
from fastapi import FastAPI

app = FastAPI()

client = TestClient(app)

def test_predict_1():
    image_path = os.path.join(os.path.dirname(__file__), 'trial.jpg')
    with open(image_path, "rb") as file:
        response = client.post("/predict/", files={"file": file})
    assert response.status_code == 200
    assert response.json() != {}

def test_predict_2():
    image_path = os.path.join(os.path.dirname(__file__), 'trial.jpg')
    with open(image_path, "rb") as file:
        response = client.post("/predict/", files={"file": file})
    assert response.json() != {}

if __name__ == "__main__":
    test_predict_1()
    test_predict_2()
