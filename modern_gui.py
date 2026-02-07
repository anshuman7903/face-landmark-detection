import customtkinter as ctk
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Face Landmark & Emotion Detection System")
app.geometry("700x600")

title = ctk.CTkLabel(
    app, 
    text="FACE LANDMARK & EMOTION ANALYZER",
    font=ctk.CTkFont(size=28, weight="bold"),
    text_color="cyan"
)
title.pack(pady=40)

# -----------------------------
# POPUP FOR LANDMARK OPTIONS
# -----------------------------
def start_landmark():
    popup = ctk.CTkToplevel(app)
    popup.title("Choose Input Mode")
    popup.geometry("400x300")

    lbl = ctk.CTkLabel(
        popup,
        text="Select Landmark Detection Mode",
        font=ctk.CTkFont(size=20, weight="bold")
    )
    lbl.pack(pady=20)

    webcam_btn = ctk.CTkButton(
        popup,
        text="Use Webcam",
        width=200,
        command=lambda: os.system("python face_landmark_mediapipe.py")
    )
    webcam_btn.pack(pady=15)

    image_btn = ctk.CTkButton(
        popup,
        text="Use Image",
        width=200,
        command=lambda: os.system("python face_landmark_image.py")
    )
    image_btn.pack(pady=15)

# Emotion Detection
def start_emotion():
    os.system("python realtime_emotion_mediapipe.py")

# Train Model
def train_model():
    os.system("python train_emotion_model.py")

# Exit
def exit_app():
    app.destroy()


# Main Buttons
btn_width = 260
btn_height = 50
btn_font = ctk.CTkFont(size=18, weight="bold")

ctk.CTkButton(app, text="Facial Landmark Detection", width=btn_width,
              height=btn_height, font=btn_font, command=start_landmark).pack(pady=15)

ctk.CTkButton(app, text="Emotion Detection", width=btn_width,
              height=btn_height, font=btn_font, command=start_emotion).pack(pady=15)

ctk.CTkButton(app, text="Train Emotion Model", width=btn_width,
              height=btn_height, font=btn_font, command=train_model).pack(pady=15)

ctk.CTkButton(app, text="Exit", width=btn_width, height=btn_height,
              fg_color="#d11a2a", hover_color="#ff4252", font=btn_font,
              command=exit_app).pack(pady=30)

app.mainloop()
