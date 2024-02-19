from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load camera matrix and distortion coefficients
camera_mtx = np.load('camera_mtx.npy')
dist_coeffs = np.load('dist_coeffs.npy')
distance_to_object = 380

import cv2
import numpy as np

def measure_object_dimension(frame,camera_mtx,dist_coeffs,distance_to_object):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to close gaps in between object edges
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Filter and draw contours
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Draw bounding box around detected objects
        if len(approx) > 2:  # Condition to filter out noise and small objects
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            width_pixels = w  # Placeholder for actual dimension calculation
            height_pixels = h  # Placeholder for actual dimension calculation

            # Calculate real-world dimensions
            width_mm = (width_pixels * distance_to_object) / camera_mtx[0, 0]  # Using f_x from the camera matrix
            height_mm = (height_pixels * distance_to_object) / camera_mtx[1, 1]  # Using f_y from the camera matrix

            # Display the dimensions on the image
            cv2.putText(frame, f"Width: {width_mm:.2f} mm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Height: {height_mm:.2f} mm", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
    return frame


def gen_frames():
    cap = cv2.VideoCapture(0)  # Use the correct device index

    while True:
        ret, frame = cap.read()
        if not ret:
            print("qwertyvfghj")
            break

        # Process the frame to detect objects and measure dimensions
        print("qwerty")
        processed_frame = measure_object_dimension(frame,camera_mtx,dist_coeffs,distance_to_object)
        print("asdfgh")
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)


