import cv2
import os

emotions = ["happy", "sad", "angry", "surprise", "neutral", "disgust"]
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
    "Choose emotion:\n0-happy\n1-sad\n2-angry\n3-surprise\n4-neutral\n5-disgust\nEnter number: "
))

emotion_name = emotions[emotion_id]
folder = os.path.join(base_path, emotion_name)
count = len(os.listdir(folder))

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.putText(frame, f"Emotion: {emotion_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Emotion Images", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # Space key
        img_path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        print(f"Saved: {img_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
