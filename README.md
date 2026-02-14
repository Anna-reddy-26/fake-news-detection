# 🕵️‍♂️ Fake Detection System (FDS.AI)

> **Advanced AI-powered platform to detect fake news, manipulated images, and deepfake videos.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 📖 Overview

In an era of misinformation, **FDS.AI** provides a robust solution to verify the authenticity of digital content. Our system leverages state-of-the-art Deep Learning models to analyze text, images, and videos for signs of manipulation.

Whether it's sensationalist news articles, edited images, or deepfake videos, FDS.AI gives you a probability score and detailed analysis to help you discern truth from fiction.

## ✨ Key Features

-   **📝 Fake News Detection**: Analyzes text for sensationalism, urgency, and verifyability patterns using NLP.
-   **🖼️ Image Manipulation Detection**: Uses CNNs and error level analysis to spot inconsistent edges, noise patterns, and unnatural artifacts.
-   **🎥 Deepfake Video Detection**: employs 3D CNNs to analyze frame-to-frame consistency and motion artifacts.
-   **🔐 Secure Authentication**: User registration and login system protected by **Flask-Login** and **Firebase**.
-   **📊 Real-time Dashboard**: Track your analysis history and view system statistics.
-   **🎨 Dynamic Themes**: Switch between **Cyber (Dark)** and **Human (Light)** modes for a personalized experience.
-   **☁️ Cloud Integration**: Stores user data and analysis history securely in **Google Firebase Firestore**.

## 🛠️ Tech Stack

-   **Backend**: Flask (Python)
-   **Frontend**: HTML5, CSS3 (Custom Variables), JavaScript
-   **AI/ML**: PyTorch, Scikit-learn, OpenCV, NLTK
-   **Database**: Firebase Firestore
-   **Authentication**: Flask-Login + Werkzeug Security

## 🚀 Installation

### Prerequisites

-   Python 3.8 or higher
-   Git
-   A Firebase Project (for `serviceAccountKey.json`)

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/fake-detection-system.git
    cd fake-detection-system
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Firebase**
    -   Go to your Firebase Console -> Project Settings -> Service Accounts.
    -   Generate a new private key.
    -   Rename the downloaded file to `serviceAccountKey.json`.
    -   Place it in the root directory of the project.

5.  **Run the Application**
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:5000`.

## 📂 Project Structure

```
├── models/             # Pre-trained AI models (News, Image, Video)
├── static/             # CSS, JS, and Uploads
├── templates/          # HTML Templates (Jinja2)
├── app.py              # Main Flask Application
├── db.py               # Database Interface (Firebase)
├── requirements.txt    # Python Dependencies
└── serviceAccountKey.json # Firebase Credentials (Not in Repo)
```

## 🧠 Model Details

### News Model
-   **Type**: NLP Classifier (LinearSVC / TF-IDF)
-   **Features**: Text sensation, emotional tone, claim verification.

### Image Model
-   **Type**: CNN (Convolutional Neural Network) based on ResNet architecture.
-   **Features**: Edge detection (Canny), Noise analysis (Laplacian), Metadata consistency.

### Video Model
-   **Type**: 3D CNN / RNN
-   **Features**: Temporal consistency, facial landmark tracking, frame-by-frame anomaly detection.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

> **Note**: This project is for educational and research purposes. AI models may not be 100% accurate. Always verify important information from multiple trusted sources.
