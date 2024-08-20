import cv2
import numpy as np

def detect_reflector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    if max_contour is not None:
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), max_contour  # 중심 좌표와 컨투어 반환
    
    return None, None

# 카메라로부터 이미지 수집
cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    center, contour = detect_reflector(frame)
    
    if center is not None:
        # 반사체가 감지되면 중심에 원을 그립니다.
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"Reflector at: {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 결과 화면에 표시
    cv2.imshow('Reflector Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

