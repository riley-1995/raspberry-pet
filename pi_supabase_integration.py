# pi_upload.py (with motion detection, model inference, organized uploads)

import os
import time
import uuid
import requests
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from supabase import create_client, Client
from datetime import datetime

# ===== CONFIG =====
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-role-key"
BUCKET_NAME = "pet-media"
MODEL_PATH = "model_edgetpu.tflite"  # compiled for Coral
LABELS = ["cat_1", "cat_2", ..., "dog_50"]  # replace with your actual labels

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== INIT CORAL MODEL =====
interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[
    tflite.load_delegate('libedgetpu.so.1')
])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== HELPERS =====
def preprocess(frame):
    input_shape = input_details[0]['shape'][1:3]
    resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    return input_data

def run_model_inference(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

def capture_images_and_video():
    img_paths = []
    for i in range(3):
        path = f"frame_{i + 1}.jpg"
        os.system(f"libcamera-jpeg -o {path} --width 640 --height 480 -n")
        img_paths.append(path)
        time.sleep(0.2)

    vid_path = "vid.mp4"
    os.system(f"libcamera-vid -t 3000 -o clip.h264 -n")
    os.system(f"MP4Box -add clip.h264 {vid_path} >/dev/null 2>&1")
    return img_paths, vid_path

def upload_file(local_path: str, storage_path: str, content_type: str):
    with open(local_path, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(storage_path, f, {"content-type": content_type})
    return supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)

def send_to_db(label_idx: int, label_name: str, image_url: str, video_url: str):
    data = {
        "ml_label_idx": label_idx,
        "ml_label_str": label_name,
        "image": image_url,
        "alt_text": f"Detected pet: {label_name} at {datetime.now().isoformat()}"
    }
    supabase.table("SpottedPiShot").insert(data).execute()

# ===== MOTION DETECTION SETUP =====
def detect_motion(prev_gray, current_gray, threshold=5000):
    diff = cv2.absdiff(prev_gray, current_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = cv2.countNonZero(thresh)
    return non_zero_count > threshold

# ===== MAIN LOOP =====
def start_detection_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detect_motion(prev_gray, gray):
            print("🚨 Motion detected!")
            softmax_output = run_model_inference(frame)
            label_idx = int(np.argmax(softmax_output))
            label_name = LABELS[label_idx]
            uid = uuid.uuid4().hex
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = f"spotted/{uid}_{label_name}_{timestamp}"

            img_paths, video_path = capture_images_and_video()
            image_urls = []
            for i, path in enumerate(img_paths):
                remote_name = f"{folder}/img{i + 1}.jpg"
                url = upload_file(path, remote_name, "image/jpeg")
                image_urls.append(url)

            video_url = upload_file(video_path, f"{folder}/vid.mp4", "video/mp4")

            send_to_db(label_idx, label_name, image_urls[0], video_url)
            print(f"✅ Uploaded {label_name} with ID {uid}")
            time.sleep(5)  # debounce

        prev_gray = gray.copy()

# Run the loop
if __name__ == "__main__":
    start_detection_loop()
