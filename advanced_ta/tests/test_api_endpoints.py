import requests

BASE_URL = "http://127.0.0.1:8000"

endpoints = [
    ("/backtest", "POST"),
    ("/optimize", "POST"),
    ("/parameters", "GET"),
    ("/stats", "GET"),
]

def test_endpoints():
    for endpoint, method in endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            if method == "GET":
                resp = requests.get(url)
            elif method == "POST":
                # Dummy payload for POST endpoints
                payload = {"parameters": {"emaPeriod": 20, "adxThreshold": 20.0, "useAdxFilter": True}}
                if endpoint == "/optimize":
                    payload["optimize_params"] = ["emaPeriod", "adxThreshold"]
                resp = requests.post(url, json=payload)
            else:
                continue
            print(f"{endpoint} [{resp.status_code}]: {resp.json()}")
        except Exception as e:
            print(f"Error accessing {endpoint}: {e}")

if __name__ == "__main__":
    test_endpoints()
