import cv2
import numpy as np
import base64
import time
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

cap = None
captured_card = None
detection_time = None
quality_message = "Waiting for ID card..."
bounding_box_found = False

def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_FOCUS, 0)

def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def calculate_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_brightness(image):
    return np.mean(image)

def calculate_contrast(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (max_val - min_val) / (max_val + min_val) if max_val + min_val != 0 else 0

def check_image_quality(image):
    """ Evaluate blur, brightness, and contrast only when ID card is detected """
    global quality_message
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = calculate_blur(gray)
    brightness_score = calculate_brightness(gray)
    contrast_score = calculate_contrast(gray)

    print(f"📊 Quality -> Blur: {blur_score:.2f}, Brightness: {brightness_score:.2f}, Contrast: {contrast_score:.2f}")

    if blur_score < 100:
        quality_message = "Too blurry, hold still!"
        return False
    if brightness_score < 100:
        quality_message = "Too dark, increase lighting!"
        return False
    if brightness_score > 200:
        quality_message = "Too bright, reduce light!"
        return False
    if contrast_score < 0.2:
        quality_message = "Low contrast, adjust camera angle!"
        return False

    quality_message = "Good quality! Capturing..."
    return True

def detect_id_card(frame):
    global captured_card, detection_time, quality_message, bounding_box_found
    bounding_box_found = False  # Reset bounding box detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 1.5 < aspect_ratio < 2.5 and cv2.contourArea(contour) > 5000:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                bounding_box_found = True  # ID card detected

                if detection_time is None:
                    detection_time = time.time()

                if time.time() - detection_time >= 3 and captured_card is None:
                    id_card = frame[y:y + h, x:x + w]

                    if not check_image_quality(id_card):
                        return frame  # Don't capture if quality is bad

                    _, buffer = cv2.imencode('.jpg', id_card, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    captured_card = base64.b64encode(buffer).decode('utf-8')

                    stop_camera()  # Turn off webcam after capture

    return frame

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_captured_card', methods=['GET'])
def get_captured_card():
    if captured_card:
        return jsonify({"id_card": captured_card})
    else:
        return jsonify({"error": "No ID card captured yet"})

@app.route('/retake', methods=['POST'])
def retake():
    global captured_card, detection_time
    captured_card = None
    detection_time = None
    start_camera()
    return jsonify({"message": "Retake triggered"})

@app.route('/get_quality_feedback', methods=['GET'])
def get_quality_feedback():
    if bounding_box_found:
        return jsonify({"message": quality_message})
    else:
        return jsonify({"message": "Waiting for ID card..."})

def generate_frames():
    start_camera()
    while True:
        if cap is None:
            break

        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_id_card(frame)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
