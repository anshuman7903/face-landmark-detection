import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

img_path = input("Enter image path: ")

image = cv2.imread(img_path)
if image is None:
    print("Error: Could not read image.")
    exit()

h, w = image.shape[:2]

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
        exit()

    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

cv2.imshow("Facial Landmarks (Image)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
