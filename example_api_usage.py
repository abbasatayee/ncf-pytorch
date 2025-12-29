"""
Example script demonstrating how to use the NCF Inference API.

Make sure the API server is running before executing this script:
    python inference.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(json.dumps(response, indent=2))

def main():
    """Example usage of the NCF Inference API"""
    
    # 1. Health Check
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response("Health Check Response", response.json())
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure the server is running:")
        print("   python inference.py")
        return
    
    # 2. Single Prediction
    print("\n2. Making a single prediction...")
    prediction_request = {
        "user_id": 0,
        "item_id": 100
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        json=prediction_request
    )
    print_response("Single Prediction Response", response.json())
    
    # 3. Batch Prediction
    print("\n3. Making batch predictions...")
    batch_request = {
        "user_id": 0,
        "item_ids": [100, 200, 300, 400, 500]
    }
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch_request
    )
    print_response("Batch Prediction Response", response.json())
    
    # 4. Get Recommendations
    print("\n4. Getting top-K recommendations...")
    recommendation_request = {
        "user_id": 0,
        "k": 10
    }
    response = requests.post(
        f"{BASE_URL}/recommend",
        json=recommendation_request
    )
    print_response("Recommendations Response", response.json())
    
    # 5. Get Recommendations with Candidate Items
    print("\n5. Getting recommendations from candidate items...")
    recommendation_request = {
        "user_id": 0,
        "k": 5,
        "candidate_item_ids": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    response = requests.post(
        f"{BASE_URL}/recommend",
        json=recommendation_request
    )
    print_response("Recommendations (with candidates) Response", response.json())
    
    print("\n" + "="*70)
    print("✓ All API examples completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()

