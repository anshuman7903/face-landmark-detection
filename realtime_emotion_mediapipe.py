import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("my_emotion_model.h5")

emotions = ["happy", "sad", "angry", "surprise", "neutral", "disgust"]

# Load Mediapipe
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for lm in result.multi_face_landmarks:
            h, w, c = frame.shape

            # Bounding box
            xs = [int(pt.x * w) for pt in lm.landmark]
            ys = [int(pt.y * h) for pt in lm.landmark]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48,48))
                gray = gray.reshape(1,48,48,1) / 255.0

                pred = np.argmax(model.predict(gray, verbose=0))
                emotion = emotions[pred]

                cv2.putText(frame, emotion, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            mp_draw.draw_landmarks(frame, lm, mp_face.FACEMESH_TESSELATION)

    cv2.imshow("Emotion + Mediapipe Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
