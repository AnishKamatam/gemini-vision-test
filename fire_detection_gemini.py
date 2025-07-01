import cv2
import requests
import base64
from io import BytesIO
from PIL import Image
import time
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

# Function to encode image as base64
def encode_image_to_base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to send frame to Gemini Vision
def check_fire_in_frame(frame):
    img_b64 = encode_image_to_base64(frame)
    data = {
        "contents": [
            {
                "parts": [
                    {"text": "Is there fire in this image? Answer only yes or no."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{ENDPOINT}?key={API_KEY}", headers=headers, json=data)
    if response.status_code == 200:
        try:
            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            return "yes" in answer
        except Exception as e:
            print("Error parsing response:", e)
            print(response.text)
            return False
    else:
        print("API Error:", response.status_code, response.text)
        return False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press 'q' to quit.")
    last_check = 0
    check_interval = 3  # seconds between API calls
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Webcam - Fire Detection', frame)
        now = time.time()
        if now - last_check > check_interval:
            print("Checking for fire...")
            fire_detected = check_fire_in_frame(frame)
            if fire_detected:
                print("🔥 FIRE DETECTED! 🔥")
            else:
                print("No fire detected.")
            last_check = now
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 