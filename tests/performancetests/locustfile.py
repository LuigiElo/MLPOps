import random
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(0.5, 10)

    @task
    def post_image(self):
        print("MyUser is posting an image with the API")
        item_id = random.randint(1, 10)

        response = self.client.post("/predict")
        print("Response status code: ", response.status_code)
        print("Response text: ", response.text)

        with self.client.post("/predict", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.failure("404 - Bad request")
            elif response.status_code == 500:
                response.failure("500 - Bad request")


# For testing add: export MYENDPOINT=http://127.0.0.1:8000
# To run in terminal: locust -f tests/performancetests/locustfile.py \
# --headless --users 10 --spawn-rate 1 --run-time 1m --host $MYENDPOINT
