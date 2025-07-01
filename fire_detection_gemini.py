import cv2
import requests
import base64
from io import BytesIO
from PIL import Image
import time
from dotenv import load_dotenv
import os
import re

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

# Function to encode image as base64
def encode_image_to_base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to send frame to Gemini Vision and get bounding box
# Returns (fire_detected, bbox) where bbox is (x1, y1, x2, y2) or None
def check_fire_and_bbox_in_frame(frame):
    img_b64 = encode_image_to_base64(frame)
    prompt = (
        "Is there fire in this image? If yes, respond with 'yes' and the bounding box coordinates as [x1, y1, x2, y2] "
        "where (x1, y1) is the top-left and (x2, y2) is the bottom-right of the fire region in pixel coordinates. "
        "If no fire, respond with 'no'. Example: yes [100, 50, 200, 150] or no."
    )
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
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
            if answer.startswith("no"):
                return False, None
            # Try to extract bounding box
            match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", answer)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return True, (x1, y1, x2, y2)
            else:
                return True, None
        except Exception as e:
            print("Error parsing response:", e)
            print(response.text)
            return False, None
    else:
        print("API Error:", response.status_code, response.text)
        return False, None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press 'q' to quit.")
    last_check = 0
    check_interval = 3  # seconds between API calls
    bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        display_frame = frame.copy()
        now = time.time()
        if now - last_check > check_interval:
            print("Checking for fire and bounding box...")
            fire_detected, bbox = check_fire_and_bbox_in_frame(frame)
            if fire_detected:
                print("🔥 FIRE DETECTED! 🔥")
                if bbox:
                    print(f"Bounding box: {bbox}")
                else:
                    print("No bounding box returned.")
            else:
                print("No fire detected.")
            last_check = now
        # Draw bounding box if available
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, 'FIRE', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imshow('Webcam - Fire Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 