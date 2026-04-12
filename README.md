# Face Landmark & Emotion Detection System

A modern Python application to perform Real-Time Face Landmark projection and Emotion Detection using TensorFlow and MediaPipe. 

## Features
- **Smart Data Collection:** Automatically crops your face using MediaPipe to keep training data purely facial geometry.
- **Deep CNN Architecture:** Trained on an optimized deep network designed for complex emotion datasets.
- **Custom Modern GUI:** Everything is run asynchronously through an interactive `customtkinter` dashboard.

---

## 🚀 Setup Guide for VS Code

If you received this project and want to run it flawlessly in VS Code, follow these steps exactly:

### 1. Prerequisites
- Have **Python (3.9 - 3.11 recommended)** installed on your machine.
- Open the root folder (`FaceLandmarkDetection`) directly inside **VS Code**.

### 2. Create the Virtual Environment
To prevent library conflicts, create an isolated virtual environment (`.venv`). Open your VS Code terminal (`Terminal -> New Terminal`) and run:
```bash
python -m venv .venv
```

### 3. Activate the Environment
You **must** activate the environment before installing packages.
- **Windows (PowerShell):** 
  ```bash
  .\.venv\Scripts\Activate
  ```
  *(If you receive an execution policy security error, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` first).*

- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```
*(You will know it worked when you see `(.venv)` pop up to the left of your terminal typing line).*

### 4. Install the AI Dependencies
With `(.venv)` activated, install the exact packages needed to run the Neural Network models effortlessly:
```bash
pip install -r requirements.txt
```

### 5. Setup the Dataset (For Training)
If you wish to train the model yourself, the AI uses standard `48x48` pixel images. 
- You can either run the `emotion_capture.py` tool to snap pictures of yourself using your webcam.
- **OR (Recommended)**: Download the professional **FER-2013 Face Dataset** from Kaggle, extract it, and simply drop the respective emotion folders directly into the `dataset/` directory.

### 6. Launch the App
Start the unified control panel by running:
```bash
python modern_gui.py
```
From the GUI dashboard, you can retrain new models, test images, or launch full live-webcam emotion tracking!
