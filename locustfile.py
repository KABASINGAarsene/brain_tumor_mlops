from locust import HttpUser, task, between

class BrainTumorTester(HttpUser):
    # Fake customers will wait between 1 to 3 seconds before sending another request
    wait_time = between(1, 3)

    @task
    def test_predict_endpoint(self):
        """
        Simulates a user uploading an MRI scan for diagnosis.
        Ensure you have an image at this path to test with!
        """
        # Change this path if your test image is located somewhere else
        image_path = "data/test/yes/Y180.jpg" 
        
        try:
            with open(image_path, "rb") as image_file:
                # Sends the POST request to the FastAPI waitstaff
                self.client.post("/predict", files={"file": image_file})
        except FileNotFoundError:
            print(f"Error: Could not find test image at {image_path}")