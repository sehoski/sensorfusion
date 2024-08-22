import cv2
import numpy as np
import matplotlib.pyplot as plt

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
            return (cx, cy), max_contour
    
    return None, None

# 내부 캘리브레이션 결과 로드
calibration_data = np.load('/home/seho/ros2_ws/src/my_camera_pkg/my_camera_pkg/calibration_result.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# 카메라로부터 이미지 수집
cap = cv2.VideoCapture(4)

# 반사체 위치를 저장할 리스트
detected_positions = []

# 실시간 그래프 설정
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
plt.xlabel('X Position')
plt.ylabel('Y Position')

# 새 카메라 매트릭스 초기화
new_camera_matrix = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 이미지 왜곡 보정
    if new_camera_matrix is None:
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    center, contour = detect_reflector(undistorted_frame)
    
    if center is not None:
        detected_positions.append(center)
        
        # 반사체가 감지되면 중앙에 원을 그리고, 컨투어를 표시합니다.
        cv2.circle(undistorted_frame, center, 5, (0, 255, 0), -1)
        cv2.drawContours(undistorted_frame, [contour], -1, (0, 255, 0), 2)
        
        # 바운딩 박스 그리기
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(undistorted_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.putText(undistorted_frame, f"Reflector at: {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 실시간 그래프 업데이트
        sc.set_offsets(detected_positions)
        ax.set_title(f"Real-time Reflector Positions ({len(detected_positions)} points)")
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # 결과 화면에 표시
    cv2.imshow('Reflector Detection', undistorted_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
