import cv2
import mediapipe as mp
import numpy as np
import keyboard
 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
draw_landmarks = False 
smile_description = {
    (1.7, 1.83) : 'Subtle smile',
    (1.83, 1.94) : 'Medium smile',
    (1.94, 2.1) : 'Broad smile',
    (2.1, 5.0) : 'Huge smile'
    
}


def get_smile_description(smile_ratio) :
    for(min_ratio, max_ratio ), description in smile_description.items():
        if min_ratio <= smile_ratio < max_ratio :
            return description
    return 'No smile'    
 
 
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
 
cap = cv2.VideoCapture(0)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if draw_landmarks :
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )
 
            top_lip = [61, 40, 37, 0, 267, 270, 269, 409, 291]
            bottom_lip = [146, 91, 181, 84, 17, 314, 405, 321, 375]
            left_corner = 61
            right_corner = 291
            left_nose = 131
            right_nose = 360
 
            top_lip_coords = [np.array([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]) for i in top_lip]
            bottom_lip_coords = [np.array([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]) for i in bottom_lip]
            left_corner_coords = np.array([face_landmarks.landmark[left_corner].x, face_landmarks.landmark[left_corner].y])
            right_corner_coords = np.array([face_landmarks.landmark[right_corner].x, face_landmarks.landmark[right_corner].y])
            left_nose_coords = np.array([face_landmarks.landmark[left_nose].x, face_landmarks.landmark[left_nose].y])
            right_nose_coords = np.array([face_landmarks.landmark[right_nose].x, face_landmarks.landmark[right_nose].y])
 
           
            horizontal_distance = euclidean_distance(left_corner_coords, right_corner_coords)
            vertical_distance = np.mean([euclidean_distance(top, bottom) for top, bottom in zip(top_lip_coords, bottom_lip_coords)])
            nose_distance = euclidean_distance(left_nose_coords, right_nose_coords)
            smile_ratio = horizontal_distance / nose_distance
            print(smile_ratio)
            if keyboard.is_pressed('z') :
                draw_landmarks = not draw_landmarks
            cv2.putText(frame, get_smile_description(smile_ratio) + ' detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Toggle mesh - z', (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Face Mesh', frame)
 
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 

cap.release()
cv2.destroyAllWindows()