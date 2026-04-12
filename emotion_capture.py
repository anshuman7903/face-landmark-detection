import cv2
import os
import mediapipe as mp

emotions = ["angry", "disgust", "happy", "neutral", "sad", "surprise"]
base_path = "dataset"

# Create folder structure
for emo in emotions:
    os.makedirs(os.path.join(base_path, emo), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Emotion Dataset Capture")
print("------------------------")
print("Press SPACE to capture an image")
print("Press Q to quit\n")

emotion_id = int(input(
    "Choose emotion:\n0-angry\n1-disgust\n2-happy\n3-neutral\n4-sad\n5-surprise\nEnter number: "
))

emotion_name = emotions[emotion_id]
folder = os.path.join(base_path, emotion_name)
count = len(os.listdir(folder))

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    face = None
    if result.multi_face_landmarks:
        for lm in result.multi_face_landmarks:
            h, w, c = frame.shape
            xs = [int(pt.x * w) for pt in lm.landmark]
            ys = [int(pt.y * h) for pt in lm.landmark]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            face = frame[y1:y2, x1:x2]
            
            # Draw rectangle to guide user
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            break

    cv2.putText(frame, f"Emotion: {emotion_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Emotion Images", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # Space key
        if face is not None and face.size != 0:
            img_path = os.path.join(folder, f"{count}.jpg")
            cv2.imwrite(img_path, face)
            count += 1
            print(f"Saved: {img_path} (Cropped Face)")
        else:
            print("No face detected! Cannot save.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
