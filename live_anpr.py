import cv2
import pytesseract
import numpy as np
import imutils
import time
import os
import sys

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not os.path.exists(TESSERACT_PATH):
    print(f"[ERROR] Tesseract not found at {TESSERACT_PATH}")
    sys.exit(1)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        return edged, gray
    except Exception as e:
        return None, None

def find_plate_contour(edged):
    try:
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                return approx
        return None
    except:
        return None

def extract_plate(image, screenCnt):
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [screenCnt], -1, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = cv2.boundingRect(screenCnt)
        cropped = image[y:y + h, x:x + w]
        return cropped if cropped.size else None
    except:
        return None

def perform_ocr(plate_image):
    try:
        plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = '-l eng --oem 1 --psm 7'
        text = pytesseract.image_to_string(plate_thresh, config=config)
        return text.strip()
    except:
        return ""

def main():
    print("[INFO] Starting ANPR system (continuous feed)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access the camera.")
        return

    last_detection_time = 0
    cooldown_seconds = 5
    last_plate_text = ""

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        edged, _ = preprocess_image(frame)
        if edged is None:
            continue

        plate_contour = find_plate_contour(edged)
        current_time = time.time()

        if plate_contour is not None and (current_time - last_detection_time) > cooldown_seconds:
            plate_img = extract_plate(frame, plate_contour)
            if plate_img is not None:
                plate_text = perform_ocr(plate_img)
                if plate_text and plate_text != last_plate_text:
                    last_detection_time = current_time
                    last_plate_text = plate_text
                    cv2.drawContours(frame, [plate_contour], -1, (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    print(f"[INFO] Detected Plate: {plate_text}")
        cv2.imshow("ANPR - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting ANPR system...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
