from locust import HttpUser, task, between
import json

class ChatUser(HttpUser):
    # Simulate users waiting between 1 to 5 seconds before sending their next message
    wait_time = between(1, 5)

    @task
    def test_chat_endpoint(self):
        # We simulate a typical payload sent to the /api/chat endpoint
        # For a true benchmark, we pass in a session_id to maintain state,
        # but to test raw load, we can generate a new random session per user
        payload = {
            "session_id": f"benchmark_user_{self.environment.runner.user_count}",
            "message": "I'm feeling really stressed about my work and I haven't been sleeping well."
        }
        
        headers = {'Content-Type': 'application/json'}
        
        # We use a post request to the endpoint
        with self.client.post("/api/chat", data=json.dumps(payload), headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
