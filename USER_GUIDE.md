# Fake Detection System - User Guide

## 1. How It Works (The Technology)
This system uses **Artificial Intelligence (AI)** to detect manipulated content. Here is what happens behind the scenes when you click "Analyze":

### 📰 Fake News Detection
- **Input**: Text content (headlines or articles).
- **Process**:
    1.  **Vectorization**: Converts words into numbers using TF-IDF (Term Frequency-Inverse Document Frequency).
    2.  **Model**: A "Passive Aggressive Classifier" compares the text against thousands of known fake and real articles.
    3.  **Heuristics**: It also checks for "shouting" (ALL CAPS) and sensational words (e.g., "SHOCKING", "UNBELIEVABLE").
- **Output**: A probability score (e.g., 94% Fake).

### 🖼️ Fake Image Detection
- **Input**: Image files (JPG, PNG).
- **Process**:
    1.  **features**: The system looks at pixel patterns.
    2.  **ELA (Error Level Analysis)**: It checks if parts of the image have different compression levels (a sign of Photoshop/editing).
    3.  **CNN (Convolutional Neural Network)**: A deep learning model scans for visual inconsistencies invisible to the human eye.
- **Output**: "Real" or "Fake" classification with a confidence bar.

### 🎥 Fake Video Detection
- **Input**: Video files (MP4, AVI).
- **Process**:
    1.  **Frame Extraction**: It splits the video into individual images (frames).
    2.  **Temporal Consistency**: It checks if objects move naturally from one frame to the next.
    3.  **Face Analysis**: It looks for "glitches" often found in Deepfakes (like non-blinking eyes or blurry face edges).
- **Output**: A comprehensive risk level.

### ☁️ Database Integration (Firebase)
- Every time you run an analysis, the result is securely saved to **Google Firebase**.
- This allows you to view your **History** and see global statistics on the **Dashboard**.

---

## 2. How to Use the Website

### Step 1: specifics
- Go to the **Home Page**.
- You will see the **Dashboard** showing total analyses and accuracy rates.
- These numbers are real and update automatically as you use the tool.

### Step 2: Detection
1.  Click **"Detect"** in the navigation bar.
2.  **Select Media Type**:
    - Click **News** for text.
    - Click **Image** for photos.
    - Click **Video** for clips.
3.  **Upload/Enter Content**:
    - For News: Paste the article text.
    - For Image/Video: Drag and drop your file or click to browse.
4.  Click **"Analyze Content"**.

### Step 3: View Results
- You will be taken to a **Results Page**.
- **Green Badge**: Likely Real.
- **Red Badge**: Likely Fake (High Risk).
- **Confidence**: Shows how sure the AI is (e.g., 98%).
- **Explanation**: A brief text explaining why it thinks so.

### Step 4: Check History
- Click **"History"** in the menu.
- You will see a list of all your past checks.
- You can review what you scanned and what the result was.

---

## 3. Troubleshooting
- **Result not saving?**
    - Ensure your internet is connected (for Firebase).
    - If you are a developer, check the `serviceAccountKey.json`.
- **Upload failed?**
    - Ensure images are under 10MB and videos are under 50MB (for smoother performance).
    - Supported formats: JPG, PNG, MP4, AVI.
