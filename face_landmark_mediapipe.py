import cv2
import mediapipe as mp
import time
import os

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Drawing style (landmarks + connections)
face_drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
conn_drawing_spec = mp_draw.DrawingSpec(thickness=1, color=(0,255,255))

def detect_landmarks(face_mesh, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    count = 0
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_draw.draw_landmarks(
                image,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=face_drawing_spec,
                connection_drawing_spec=conn_drawing_spec
            )
            count += 1
    return image, count

print("✅ Mediapipe Face Landmark System Started")
print("1. Detect from Image")
print("2. Detect from Webcam")
mode = input("Enter 1 or 2: ").strip()

if mode == "1":
    path = input("Enter image path: ")
    if not os.path.exists(path):
        print("❌ Image not found!")
        exit()
    image = cv2.imread(path)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=3) as face_mesh:
        result, count = detect_landmarks(face_mesh, image)
    print(f"✅ Faces detected: {count}")
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/output_image.jpg", result)
    cv2.imshow("Landmark Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "2":
    cap = cv2.VideoCapture(0)
    prev = 0
    with mp_face.FaceMesh(max_num_faces=3, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        print("🎥 Webcam ON — Press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            now = time.time()
            fps = 1 / (now - prev) if prev else 0
            prev = now
            result, count = detect_landmarks(face_mesh, frame)
            cv2.putText(result, f"Faces: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(result, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Mediapipe Face Mesh (468 Points)", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Invalid choice.")
