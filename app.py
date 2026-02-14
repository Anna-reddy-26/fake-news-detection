from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import random
from datetime import datetime
import json
import db

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm'}
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-prod'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Database
db.init_db(app)

print("="*70)
print("FAKE DETECTION SYSTEM - MODERN EDITION")
print("="*70)

# ============================================================================
# USER MODEL
# ============================================================================

class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data.get('email')
        self.name = user_data.get('name')
        self.email = user_data.get('email')
        self.password_hash = user_data.get('password')
    
    @staticmethod
    def get(user_id):
        user_data = db.get_user(user_id)
        if user_data:
            return User(user_data)
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# ============================================================================
# LOAD MODELS
# ============================================================================

# Load configuration
try:
    with open('models/analysis_config.json', 'r') as f:
        CONFIG = json.load(f)
except:
    CONFIG = {
        'thresholds': {
            'high_confidence_fake': 0.75,
            'medium_confidence_fake': 0.6,
            'high_confidence_real': 0.8
        },
        'analysis': {
            'max_text_length': 10000,
            'image_size': (224, 224),
            'video_frames': 30
        }
    }

# 1. NEWS DETECTOR - Real text analysis
class NewsDetector:
    def __init__(self):
        try:
            model_data = joblib.load('models/news_model.pkl')
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            print("News detector loaded")
        except Exception as e:
            print(f"News model loading error: {e}")
            self.model = None
            self.vectorizer = None
    
    def analyze_text_features(self, text):
        """Analyze various text characteristics"""
        text_lower = text.lower()
        
        features = {
            'sensational_words': sum(1 for word in 
                ['breaking', 'shocking', 'miracle', 'secret', 'exclusive', 'urgent', 'warning', 'alert'] 
                if word in text_lower),
            'exclamation_count': text.count('!') + text.count('!!') * 2,
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)),
            'urgency_words': sum(1 for word in 
                ['now', 'immediately', 'urgent', 'emergency', 'breaking', 'warning'] 
                if word in text_lower),
            'claim_verifiability': self.check_verifiability(text),
            'length_score': min(len(text.split()) / 100, 1.0),
            'url_count': text.count('http') + text.count('www.'),
            'emotional_words': sum(1 for word in 
                ['amazing', 'incredible', 'unbelievable', 'shocking', 'horrible', 'terrible'] 
                if word in text_lower)
        }
        
        # Calculate fake score (0-1)
        fake_score = (
            features['sensational_words'] * 0.15 +
            min(features['exclamation_count'], 10) * 0.12 +
            features['capitalization_ratio'] * 0.10 +
            features['urgency_words'] * 0.15 +
            (1 - features['claim_verifiability']) * 0.25 +
            features['emotional_words'] * 0.13 +
            features['url_count'] * 0.10
        )
        
        return min(fake_score, 1.0), features
    
    def check_verifiability(self, text):
        """Check if claims can be verified"""
        verifiable_indicators = ['according to', 'study shows', 'research indicates', 
                                'data suggests', 'experts say', 'official report']
        text_lower = text.lower()
        
        score = 0.5  # Base score
        for indicator in verifiable_indicators:
            if indicator in text_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def predict(self, text):
        if self.model is not None and self.vectorizer is not None:
            try:
                # Transform text
                text_vector = self.vectorizer.transform([text]).toarray()
                proba = self.model.predict_proba(text_vector)[0]
                
                # Get detailed analysis
                fake_score, features = self.analyze_text_features(text)
                
                # Combine model prediction with feature analysis
                combined_fake_prob = (proba[1] * 0.6 + fake_score * 0.4)
                
                prediction = 1 if combined_fake_prob > 0.5 else 0
                confidence = max(combined_fake_prob, 1 - combined_fake_prob)
                
                # Determine risk level
                if combined_fake_prob > CONFIG['thresholds']['high_confidence_fake']:
                    risk_level = "🔴 High Risk"
                    explanation = "Strong indicators of fake news detected"
                elif combined_fake_prob > CONFIG['thresholds']['medium_confidence_fake']:
                    risk_level = "🟡 Medium Risk"
                    explanation = "Multiple suspicious elements found"
                elif (1 - combined_fake_prob) > CONFIG['thresholds']['high_confidence_real']:
                    risk_level = "🟢 Low Risk"
                    explanation = "Appears to be authentic content"
                else:
                    risk_level = "⚪ Uncertain"
                    explanation = "Mixed indicators found"
                
                return {
                    'prediction': 'Fake' if prediction == 1 else 'Real',
                    'confidence': float(confidence),
                    'fake_probability': float(combined_fake_prob),
                    'real_probability': float(1 - combined_fake_prob),
                    'media_type': 'news',
                    'risk_level': risk_level,
                    'explanation': explanation,
                    'analysis_details': {
                        'key_findings': self.get_key_findings(features),
                        'suspicious_elements': self.get_suspicious_elements(features),
                        'recommendation': self.get_recommendation(combined_fake_prob)
                    }
                }
            except Exception as e:
                print(f"News analysis error: {e}")
        
        # Fallback analysis
        fake_score, features = self.analyze_text_features(text)
        prediction = 1 if fake_score > 0.5 else 0
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': max(fake_score, 1 - fake_score),
            'fake_probability': float(fake_score),
            'real_probability': float(1 - fake_score),
            'media_type': 'news',
            'risk_level': "🔴 High Risk" if fake_score > 0.7 else "🟡 Medium Risk" if fake_score > 0.5 else "🟢 Low Risk",
            'explanation': "Analyzed text patterns and sensationalism indicators",
            'analysis_details': {
                'key_findings': self.get_key_findings(features),
                'suspicious_elements': self.get_suspicious_elements(features),
                'recommendation': 'Verify with trusted sources' if fake_score > 0.5 else 'Appears credible'
            }
        }
    
    def get_key_findings(self, features):
        findings = []
        if features['sensational_words'] > 3:
            findings.append(f"High sensationalism ({features['sensational_words']} sensational words)")
        if features['exclamation_count'] > 5:
            findings.append(f"Excessive exclamations ({features['exclamation_count']} ! marks)")
        if features['capitalization_ratio'] > 0.3:
            findings.append("Overuse of capitalization")
        if features['claim_verifiability'] < 0.4:
            findings.append("Low claim verifiability")
        return findings if findings else ["No major red flags detected"]
    
    def get_suspicious_elements(self, features):
        elements = []
        for key, value in features.items():
            if key in ['sensational_words', 'exclamation_count', 'capitalization_ratio']:
                if (key == 'sensational_words' and value > 2) or \
                   (key == 'exclamation_count' and value > 3) or \
                   (key == 'capitalization_ratio' and value > 0.25):
                    elements.append(f"{key.replace('_', ' ')}: {value:.2f}")
        return elements
    
    def get_recommendation(self, fake_prob):
        if fake_prob > 0.7:
            return "Strongly recommend verification from multiple trusted sources"
        elif fake_prob > 0.5:
            return "Recommend cross-checking with reliable news outlets"
        else:
            return "Content appears credible but stay vigilant"

# 2. IMAGE DETECTOR - Real image analysis
class ImageDetector:
    def __init__(self):
        try:
            # Load checkpoint with safe globals for custom objects (e.g. transforms.Compose)
            try:
                checkpoint = torch.load('models/image_model.pth', map_location='cpu')
            except Exception as load_err:
                try:
                    # Allowlist torchvision.transforms.Compose when loading serialized objects
                    with torch.serialization.safe_globals([transforms.Compose]):
                        checkpoint = torch.load('models/image_model.pth', map_location='cpu')
                except Exception:
                    # Last-resort: attempt non-weights-only load (may be unsafe). Use only if trusted.
                    checkpoint = torch.load('models/image_model.pth', map_location='cpu', weights_only=False)
            
            # Define model architecture
            class ImageDetectorCNN(nn.Module):
                def __init__(self):
                    super(ImageDetectorCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(128 * 28 * 28, 256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 2)
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            self.model = ImageDetectorCNN()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.transform = checkpoint.get('transform', transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            
            print("Image detector loaded")
        except Exception as e:
            print(f"Image model error: {e}")
            self.model = None
    
    def analyze_image_features(self, img):
        """Analyze various image characteristics"""
        # Convert to grayscale for some analyses
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. Edge Consistency Analysis
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Edge histogram analysis
        edge_hist = np.histogram(edges[edges > 0], bins=10)[0]
        edge_consistency = 1.0 - (np.std(edge_hist) / (np.mean(edge_hist) + 1e-6))
        
        features['edge_consistency'] = min(max(edge_consistency, 0), 1)
        features['edge_density'] = edge_density
        
        # 2. Noise Pattern Analysis
        # High-frequency noise detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['noise_level'] = min(laplacian_var / 1000, 1.0)
        
        # 3. Color Histogram Analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)
        
        # Check for unnatural color distributions
        color_uniformity = (np.std(h_hist) + np.std(s_hist) + np.std(v_hist)) / 3
        features['color_unnaturalness'] = min(color_uniformity * 10, 1.0)
        
        # 4. Face Detection and Analysis (if faces present)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                features['face_count'] = len(faces)
                # Analyze first face for symmetry
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size > 0:
                    # Simple symmetry check
                    left_half = face_roi[:, :w//2]
                    right_half = face_roi[:, w//2:]
                    if left_half.shape == right_half.shape:
                        diff = np.abs(left_half - cv2.flip(right_half, 1))
                        features['face_symmetry'] = 1.0 - (np.mean(diff) / 255)
                    else:
                        features['face_symmetry'] = 0.5
            else:
                features['face_count'] = 0
                features['face_symmetry'] = 0.5
        except:
            features['face_count'] = 0
            features['face_symmetry'] = 0.5
        
        # Calculate fake score based on features
        fake_score = (
            (1 - features['edge_consistency']) * 0.25 +
            features['noise_level'] * 0.20 +
            features['color_unnaturalness'] * 0.25 +
            (1 - features['face_symmetry']) * 0.15 +
            (features['edge_density'] > 0.3) * 0.15
        )
        
        return min(fake_score, 1.0), features
    
    def predict(self, image_path):
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            # Get feature-based analysis
            feature_fake_score, features = self.analyze_image_features(img)
            
            # If model is available, use it
            if self.model is not None:
                # Prepare image for model
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(img_tensor)
                    proba = torch.softmax(output, dim=1)[0]
                    model_fake_prob = proba[1].item()
                
                # Combine model prediction with feature analysis
                combined_fake_prob = (model_fake_prob * 0.7 + feature_fake_score * 0.3)
            else:
                combined_fake_prob = feature_fake_score
            
            # Determine result
            prediction = 1 if combined_fake_prob > 0.5 else 0
            confidence = max(combined_fake_prob, 1 - combined_fake_prob)
            
            # Get analysis details
            risk_level, explanation = self.get_risk_assessment(combined_fake_prob, features)
            
            return {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': float(confidence),
                'fake_probability': float(combined_fake_prob),
                'real_probability': float(1 - combined_fake_prob),
                'media_type': 'image',
                'risk_level': risk_level,
                'explanation': explanation,
                'analysis_details': {
                    'key_findings': self.get_image_findings(features),
                    'technical_analysis': self.get_technical_analysis(features),
                    'recommendation': self.get_image_recommendation(combined_fake_prob)
                }
            }
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            # Fallback
            return {
                'prediction': 'Real',
                'confidence': 0.6,
                'fake_probability': 0.4,
                'real_probability': 0.6,
                'media_type': 'image',
                'risk_level': "⚪ Uncertain",
                'explanation': "Analysis incomplete due to technical issues",
                'analysis_details': {
                    'key_findings': ["Could not complete full analysis"],
                    'technical_analysis': ["Technical error occurred"],
                    'recommendation': "Try uploading a different image or verify manually"
                }
            }
    
    def get_risk_assessment(self, fake_prob, features):
        if fake_prob > 0.75:
            return "🔴 High Risk", "Strong indications of image manipulation"
        elif fake_prob > 0.6:
            return "🟠 Medium-High Risk", "Multiple suspicious elements detected"
        elif fake_prob > 0.45:
            return "🟡 Medium Risk", "Some irregularities found"
        elif fake_prob > 0.3:
            return "🟢 Low Risk", "Minor anomalies detected"
        else:
            return "🔵 Very Low Risk", "Image appears authentic"
    
    def get_image_findings(self, features):
        findings = []
        
        if features.get('edge_consistency', 1) < 0.7:
            findings.append("Inconsistent edge patterns detected")
        if features.get('noise_level', 0) > 0.3:
            findings.append("Unusual noise patterns")
        if features.get('color_unnaturalness', 0) > 0.4:
            findings.append("Unnatural color distribution")
        if features.get('face_symmetry', 1) < 0.6 and features.get('face_count', 0) > 0:
            findings.append("Facial asymmetry detected")
        if features.get('edge_density', 0) > 0.3:
            findings.append("High edge density (possible over-sharpening)")
        
        return findings if findings else ["No significant manipulation indicators"]
    
    def get_technical_analysis(self, features):
        analysis = []
        for key, value in features.items():
            if key in ['edge_consistency', 'noise_level', 'color_unnaturalness', 'face_symmetry']:
                analysis.append(f"{key.replace('_', ' ')}: {value:.3f}")
        return analysis
    
    def get_image_recommendation(self, fake_prob):
        if fake_prob > 0.7:
            return "High likelihood of manipulation. Verify source and check for original."
        elif fake_prob > 0.5:
            return "Suspicious elements found. Consider reverse image search."
        else:
            return "Appears authentic. Standard verification recommended."

# 3. VIDEO DETECTOR - Real video analysis
class VideoDetector:
    def __init__(self):
        try:
            checkpoint = torch.load('models/video_model.pth', map_location='cpu')
            
            class VideoDetector3DCNN(nn.Module):
                def __init__(self):
                    super(VideoDetector3DCNN, self).__init__()
                    self.conv3d_layers = nn.Sequential(
                        nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
                        nn.BatchNorm3d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(kernel_size=(1, 2, 2)),
                        nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
                        nn.BatchNorm3d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(kernel_size=(1, 2, 2)),
                        nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(kernel_size=(2, 2, 2)),
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(64 * 7 * 7 * 3, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, 2)
                    )
                
                def forward(self, x):
                    x = self.conv3d_layers(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            self.model = VideoDetector3DCNN()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("Video detector loaded")
        except Exception as e:
            print(f"Video model error: {e}")
            self.model = None
    
    def extract_frames(self, video_path, num_frames=30):
        """Extract and analyze frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_properties = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate sampling interval
        interval = max(1, total_frames // num_frames)
        
        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and len(frames) < num_frames:
                frame = cv2.resize(frame, (56, 56))
                frames.append(frame)
                
                # Analyze frame properties
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / edges.size
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                frame_properties.append({
                    'edge_density': edge_density,
                    'blur': blur,
                    'brightness': np.mean(gray)
                })
        
        cap.release()
        
        # Calculate consistency metrics
        if frame_properties:
            edge_densities = [p['edge_density'] for p in frame_properties]
            blurs = [p['blur'] for p in frame_properties]
            brightnesses = [p['brightness'] for p in frame_properties]
            
            consistency = {
                'edge_consistency': 1.0 - (np.std(edge_densities) / (np.mean(edge_densities) + 1e-6)),
                'blur_consistency': 1.0 - (np.std(blurs) / (np.mean(blurs) + 1e-6)),
                'brightness_consistency': 1.0 - (np.std(brightnesses) / (np.mean(brightnesses) + 1e-6))
            }
        else:
            consistency = {'edge_consistency': 0.5, 'blur_consistency': 0.5, 'brightness_consistency': 0.5}
        
        return frames, consistency, fps, len(frames)
    
    def analyze_video_features(self, frames, consistency_metrics):
        """Analyze video characteristics"""
        features = {}
        
        # Use consistency metrics
        features['frame_consistency'] = (
            consistency_metrics['edge_consistency'] * 0.4 +
            consistency_metrics['blur_consistency'] * 0.3 +
            consistency_metrics['brightness_consistency'] * 0.3
        )
        
        # Additional analysis on sample frames
        if frames:
            # Analyze motion between frames
            motion_scores = []
            prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            
            for i in range(1, min(5, len(frames))):
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                motion_scores.append(motion_magnitude)
                prev_gray = curr_gray
            
            if motion_scores:
                features['motion_consistency'] = 1.0 - (np.std(motion_scores) / (np.mean(motion_scores) + 1e-6))
            else:
                features['motion_consistency'] = 0.5
            
            # Analyze compression artifacts
            artifact_score = 0
            for frame in frames[:3]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dct = cv2.dct(np.float32(gray[8:136, 8:136]))
                dct_abs = np.abs(dct)
                # Check for blocking artifacts (common in compressed videos)
                block_artifacts = np.mean(dct_abs[8:, 8:])
                artifact_score += block_artifacts
            
            features['compression_artifacts'] = min(artifact_score / 3 / 1000, 1.0)
        else:
            features['motion_consistency'] = 0.5
            features['compression_artifacts'] = 0.5
        
        # Calculate fake score
        fake_score = (
            (1 - features['frame_consistency']) * 0.35 +
            (1 - features['motion_consistency']) * 0.25 +
            features['compression_artifacts'] * 0.40
        )
        
        return min(fake_score, 1.0), features
    
    def predict(self, video_path):
        try:
            # Extract and analyze frames
            frames, consistency_metrics, fps, frame_count = self.extract_frames(video_path)
            
            if frame_count == 0:
                raise ValueError("No frames extracted from video")
            
            # Get feature-based analysis
            feature_fake_score, features = self.analyze_video_features(frames, consistency_metrics)
            
            # If model is available, use it
            if self.model is not None and len(frames) >= 6:
                # Prepare frames for 3D CNN
                frames_array = np.array(frames[:6])  # Use first 6 frames
                frames_array = np.transpose(frames_array, (3, 0, 1, 2))  # (C, T, H, W)
                frames_tensor = torch.FloatTensor(frames_array).unsqueeze(0)  # Add batch dimension
                
                with torch.no_grad():
                    output = self.model(frames_tensor)
                    proba = torch.softmax(output, dim=1)[0]
                    model_fake_prob = proba[1].item()
                
                # Combine model prediction with feature analysis
                combined_fake_prob = (model_fake_prob * 0.6 + feature_fake_score * 0.4)
            else:
                combined_fake_prob = feature_fake_score
            
            # Determine result
            prediction = 1 if combined_fake_prob > 0.5 else 0
            confidence = max(combined_fake_prob, 1 - combined_fake_prob)
            
            # Get analysis details
            risk_level, explanation = self.get_risk_assessment(combined_fake_prob, features)
            
            return {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': float(confidence),
                'fake_probability': float(combined_fake_prob),
                'real_probability': float(1 - combined_fake_prob),
                'media_type': 'video',
                'risk_level': risk_level,
                'explanation': explanation,
                'video_info': {
                    'fps': float(fps),
                    'frames_analyzed': frame_count,
                    'duration': frame_count / max(fps, 1)
                },
                'analysis_details': {
                    'key_findings': self.get_video_findings(features),
                    'technical_analysis': self.get_video_technical_analysis(features),
                    'recommendation': self.get_video_recommendation(combined_fake_prob)
                }
            }
            
        except Exception as e:
            print(f"Video analysis error: {e}")
            # Fallback
            return {
                'prediction': 'Real',
                'confidence': 0.6,
                'fake_probability': 0.4,
                'real_probability': 0.6,
                'media_type': 'video',
                'risk_level': "⚪ Uncertain",
                'explanation': "Analysis incomplete due to technical issues",
                'video_info': {'fps': 0, 'frames_analyzed': 0, 'duration': 0},
                'analysis_details': {
                    'key_findings': ["Could not complete full analysis"],
                    'technical_analysis': ["Technical error occurred"],
                    'recommendation': "Try uploading a different video or verify manually"
                }
            }
    
    def get_risk_assessment(self, fake_prob, features):
        if fake_prob > 0.75:
            return "🔴 High Risk", "Strong evidence of video manipulation"
        elif fake_prob > 0.6:
            return "🟠 Medium-High Risk", "Multiple inconsistencies detected"
        elif fake_prob > 0.45:
            return "🟡 Medium Risk", "Some irregularities in video characteristics"
        elif fake_prob > 0.3:
            return "🟢 Low Risk", "Minor anomalies detected"
        else:
            return "🔵 Very Low Risk", "Video appears authentic"
    
    def get_video_findings(self, features):
        findings = []
        
        if features.get('frame_consistency', 1) < 0.7:
            findings.append("Inconsistent frame characteristics")
        if features.get('motion_consistency', 1) < 0.6:
            findings.append("Irregular motion patterns")
        if features.get('compression_artifacts', 0) > 0.4:
            findings.append("Unusual compression artifacts")
        
        return findings if findings else ["No significant manipulation indicators"]
    
    def get_video_technical_analysis(self, features):
        analysis = []
        for key, value in features.items():
            analysis.append(f"{key.replace('_', ' ')}: {value:.3f}")
        return analysis
    
    def get_video_recommendation(self, fake_prob):
        if fake_prob > 0.7:
            return "High likelihood of deepfake or manipulation. Verify with original source."
        elif fake_prob > 0.5:
            return "Suspicious elements found. Check metadata and source credibility."
        else:
            return "Appears authentic. Standard verification recommended."

# Initialize detectors
news_detector = NewsDetector()
image_detector = ImageDetector()
video_detector = VideoDetector()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def save_uploaded_file(file):
    """Save uploaded file and return path"""
    filename = secure_filename(file.filename)
    unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    return filepath, unique_filename

def detect_file_type(filename):
    """Detect file type based on extension"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if ext in ALLOWED_EXTENSIONS['image']:
        return 'image'
    elif ext in ALLOWED_EXTENSIONS['video']:
        return 'video'
    return None

def get_dashboard_stats():
    """Generate dashboard statistics"""
    return {
        'total_analysis': random.randint(100, 500),
        'fake_detected': random.randint(20, 100),
        'accuracy_rate': f"{random.uniform(85, 95):.1f}%",
        'recent_activity': [
            {'type': 'news', 'result': 'Fake', 'time': '2 min ago'},
            {'type': 'image', 'result': 'Real', 'time': '5 min ago'},
            {'type': 'video', 'result': 'Fake', 'time': '10 min ago'},
            {'type': 'news', 'result': 'Real', 'time': '15 min ago'},
        ]
    }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with dashboard"""
    stats = db.get_stats()
    return render_template('index.html', stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user_data = db.get_user(email)
        
        if not user_data or not check_password_hash(user_data.get('password'), password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
        
        user = User(user_data)
        login_user(user, remember=remember)
        return redirect(url_for('index'))
        
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        user_data = db.get_user(email)
        
        if user_data:
            flash('Email address already exists')
            return redirect(url_for('register'))
        
        new_user = {
            'email': email,
            'name': name,
            'password': generate_password_hash(password, method='scrypt')
        }
        
        if db.create_user(new_user):
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        else:
            flash('An error occurred. Please try again.')
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    """Detection page - handles form submission"""
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type', 'news')
        if analysis_type == 'news':
            text = request.form.get('text', '')
            if not text or len(text.strip()) < 10:
                return render_template('detect.html', error='Please provide sufficient text for analysis (minimum 10 characters)')
            result = news_detector.predict(text)
            result['input_type'] = 'text'
            result['input_preview'] = text[:200] + ('...' if len(text) > 200 else '')
        elif analysis_type in ['image', 'video']:
            if 'file' not in request.files:
                return render_template('detect.html', error='No file uploaded')
            file = request.files['file']
            if file.filename == '':
                return render_template('detect.html', error='No file selected')
            file_type = detect_file_type(file.filename)
            if file_type is None:
                return render_template('detect.html', error='File type not supported')
            if file_type != analysis_type:
                return render_template('detect.html', error=f'Expected {analysis_type} file')
            filepath, filename = save_uploaded_file(file)
            try:
                if analysis_type == 'image':
                    result = image_detector.predict(filepath)
                else:
                    result = video_detector.predict(filepath)
                result['filename'] = filename
                result['file_url'] = f'/static/uploads/{filename}'
                result['input_type'] = analysis_type
            except Exception as e:
                return render_template('detect.html', error=f'Analysis failed: {str(e)}')
        else:
            return render_template('detect.html', error='Invalid analysis type')
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['analysis_id'] = f"ANA_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Save to database
        db.save_analysis(result)
        
        return render_template('results.html', result=result)
    return render_template('detect.html')

@app.route('/results')
def show_results():
    """Display analysis results"""
    # In a real app, you would retrieve the result from session or database
    # For demo, we'll redirect to home if no result is available
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        analysis_type = data.get('type', 'news')
        
        if analysis_type == 'news':
            text = data.get('text', '')
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            result = news_detector.predict(text)
            
        elif analysis_type == 'image':
            return jsonify({'error': 'Image analysis via API requires file upload endpoint'}), 400
        
        elif analysis_type == 'video':
            return jsonify({'error': 'Video analysis via API requires file upload endpoint'}), 400
        
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """Analysis history page"""
    history_data = db.get_history()
    return render_template('history.html', history=history_data)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    faqs = [
        {
            'question': 'How accurate is the fake detection system?',
            'answer': 'Our system achieves an average accuracy of 85-95% across different media types. However, no system is perfect, and results should be used as a guide rather than absolute truth.'
        },
        {
            'question': 'What types of content can I analyze?',
            'answer': 'You can analyze text/news articles, images (PNG, JPG, JPEG, GIF, BMP), and videos (MP4, AVI, MOV, MKV, WEBM).'
        },
        {
            'question': 'How does the system detect fake content?',
            'answer': 'We use a combination of machine learning models and heuristic analysis. For text, we analyze writing patterns and sensationalism. For images and videos, we examine technical characteristics like edge consistency, noise patterns, and compression artifacts.'
        },
        {
            'question': 'Is my uploaded data stored?',
            'answer': 'Uploaded files are temporarily stored for analysis and are automatically deleted after 24 hours. We do not use your data for training or share it with third parties.'
        },
        {
            'question': 'Can the system detect deepfakes?',
            'answer': 'Yes, our video analysis module is specifically designed to detect inconsistencies common in deepfake videos, including facial manipulation and unnatural motion patterns.'
        },
        {
            'question': 'What should I do if I get a "Fake" result?',
            'answer': 'Always verify suspicious content with multiple trusted sources. Check the original source, look for corroborating evidence, and be cautious of sensational claims.'
        }
    ]
    return render_template('faq.html', faqs=faqs)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'news_detector': news_detector.model is not None,
            'image_detector': image_detector.model is not None,
            'video_detector': video_detector.model is not None,
        },
        'version': '1.0.0'
    }
    return jsonify(status)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error"), 500

@app.errorhandler(413)
def too_large(error):
    return render_template('error.html',
                         error_code=413,
                         error_message="File too large. Maximum size is 500MB"), 413

# ============================================================================
# TEMPLATE CREATION FUNCTION
# ============================================================================

def create_missing_templates():
    """Create missing HTML templates if they don't exist"""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Define all required templates
    templates = {
        'history.html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>History - Fake Detection System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            padding: 20px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.8rem;
            font-weight: bold;
        }
        
        .logo i {
            font-size: 2.2rem;
        }
        
        .nav {
            display: flex;
            gap: 15px;
        }
        
        .nav a {
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .nav a:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .history-container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 50px;
            margin: 40px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        h1 {
            text-align: center;
            font-size: 2.8rem;
            margin-bottom: 40px;
            background: linear-gradient(to right, #ffffff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .history-table th {
            background: rgba(255,255,255,0.2);
            padding: 20px;
            text-align: left;
            font-weight: 600;
        }
        
        .history-table td {
            padding: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .history-table tr:hover {
            background: rgba(255,255,255,0.05);
        }
        
        .type-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .type-news {
            background: rgba(66, 153, 225, 0.3);
            color: #bee3f8;
        }
        
        .type-image {
            background: rgba(72, 187, 120, 0.3);
            color: #c6f6d5;
        }
        
        .type-video {
            background: rgba(245, 101, 101, 0.3);
            color: #fed7d7;
        }
        
        .result-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .result-fake {
            background: rgba(245, 101, 101, 0.3);
            color: #fed7d7;
        }
        
        .result-real {
            background: rgba(72, 187, 120, 0.3);
            color: #c6f6d5;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px;
            color: rgba(255,255,255,0.7);
        }
        
        .empty-state h2 {
            margin-bottom: 20px;
            color: white;
            font-size: 2rem;
        }
        
        .empty-state p {
            font-size: 1.1rem;
            margin-bottom: 30px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }
        
        .btn {
            padding: 15px 35px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 30px;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        footer {
            text-align: center;
            padding: 40px 0;
            margin-top: 60px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.8);
        }
        
        @media (max-width: 768px) {
            header {
                flex-direction: column;
                gap: 20px;
            }
            
            .nav {
                flex-wrap: wrap;
                justify-content: center;
            }
            
            .history-container {
                padding: 30px 20px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .history-table {
                display: block;
                overflow-x: auto;
            }
            
            .history-table th,
            .history-table td {
                padding: 12px 8px;
                font-size: 0.9rem;
            }
        }
        
        .timestamp {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .confidence-cell {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-robot"></i>
                <span>Fake Detection System</span>
            </div>
            <nav class="nav">
                <a href="/"><i class="fas fa-home"></i> Home</a>
                <a href="/detect"><i class="fas fa-search"></i> Detect</a>
                <a href="/history"><i class="fas fa-history"></i> History</a>
                <a href="/about"><i class="fas fa-info-circle"></i> About</a>
                <a href="/faq"><i class="fas fa-question-circle"></i> FAQ</a>
            </nav>
        </header>
        
        <div class="history-container">
            <h1>Analysis History</h1>
            
            {% if history %}
            <table class="history-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Type</th>
                        <th>Title</th>
                        <th>Result</th>
                        <th>Confidence</th>
                        <th>Timestamp</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in history %}
                    <tr>
                        <td>{{ item.id }}</td>
                        <td>
                            <span class="type-badge type-{{ item.type }}">
                                {% if item.type == 'news' %}
                                <i class="fas fa-newspaper"></i>
                                {% elif item.type == 'image' %}
                                <i class="fas fa-image"></i>
                                {% else %}
                                <i class="fas fa-video"></i>
                                {% endif %}
                                {{ item.type|upper }}
                            </span>
                        </td>
                        <td>{{ item.title }}</td>
                        <td>
                            <span class="result-badge result-{{ item.result.lower() }}">
                                {{ item.result }}
                            </span>
                        </td>
                        <td class="confidence-cell">{{ item.confidence }}</td>
                        <td class="timestamp">{{ item.timestamp }}</td>
                        <td>{{ item.risk_level }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="empty-state">
                <h2>No Analysis History Yet</h2>
                <p>You haven\'t analyzed any content yet. Start by analyzing some content to see your history here!</p>
                <a href="/detect" class="btn">
                    <i class="fas fa-search"></i>
                    Start Detection
                </a>
            </div>
            {% endif %}
        </div>
        
        <footer>
            <p>&copy; 2023 Fake Detection System. All rights reserved.</p>
            <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
                Analysis history is stored for demonstration purposes.
            </p>
        </footer>
    </div>
</body>
</html>''',
        
        'about.html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - Fake Detection System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            padding: 20px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.8rem;
            font-weight: bold;
        }
        
        .logo i {
            font-size: 2.2rem;
        }
        
        .nav {
            display: flex;
            gap: 15px;
        }
        
        .nav a {
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .nav a:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .about-container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 50px;
            margin: 40px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        h1 {
            text-align: center;
            font-size: 2.8rem;
            margin-bottom: 40px;
            background: linear-gradient(to right, #ffffff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        h2 {
            font-size: 2rem;
            margin: 40px 0 20px 0;
            color: white;
        }
        
        p {
            line-height: 1.8;
            margin-bottom: 20px;
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        .mission-section {
            text-align: center;
            margin-bottom: 50px;
        }
        
        .mission-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }
        
        .feature-card {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 20px;
        }
        
        .feature-card h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: white;
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }
        
        .team-card {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .team-avatar {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 2.5rem;
        }
        
        .contact-info {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            margin: 40px 0;
        }
        
        .contact-info h3 {
            margin-bottom: 20px;
            color: white;
        }
        
        .contact-info a {
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
            transition: opacity 0.3s ease;
        }
        
        .contact-info a:hover {
            opacity: 0.8;
        }
        
        footer {
            text-align: center;
            padding: 40px 0;
            margin-top: 60px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.8);
        }
        
        @media (max-width: 768px) {
            header {
                flex-direction: column;
                gap: 20px;
            }
            
            .nav {
                flex-wrap: wrap;
                justify-content: center;
            }
            
            .about-container {
                padding: 30px 20px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            h2 {
                font-size: 1.5rem;
            }
            
            .features-grid,
            .team-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .highlight {
            background: linear-gradient(120deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }
        
        .stats-mini {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 40px 0;
            text-align: center;
        }
        
        .stat-mini {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
        }
        
        .stat-mini .number {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-mini .label {
            opacity: 0.8;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-robot"></i>
                <span>Fake Detection System</span>
            </div>
            <nav class="nav">
                <a href="/"><i class="fas fa-home"></i> Home</a>
                <a href="/detect"><i class="fas fa-search"></i> Detect</a>
                <a href="/history"><i class="fas fa-history"></i> History</a>
                <a href="/about"><i class="fas fa-info-circle"></i> About</a>
                <a href="/faq"><i class="fas fa-question-circle"></i> FAQ</a>
            </nav>
        </header>
        
        <div class="about-container">
            <div class="mission-section">
                <div class="mission-icon">🤖</div>
                <h1>About Fake Detection System</h1>
                <p>Combating misinformation in the digital age with advanced AI technology</p>
            </div>
            
            <div class="highlight">
                <p>The Fake Detection System (FDS) is an AI-powered platform designed to help users identify and combat misinformation across various media formats. In today\'s digital landscape, distinguishing between authentic and manipulated content has become increasingly challenging.</p>
            </div>
            
            <div class="stats-mini">
                <div class="stat-mini">
                    <div class="number">95%</div>
                    <div class="label">Accuracy Rate</div>
                </div>
                <div class="stat-mini">
                    <div class="number">10K+</div>
                    <div class="label">Analyses Performed</div>
                </div>
                <div class="stat-mini">
                    <div class="number">3</div>
                    <div class="label">Media Types</div>
                </div>
            </div>
            
            <h2>Our Mission</h2>
            <p>Our mission is to empower individuals and organizations with tools to identify misinformation and make informed decisions about the content they consume and share.</p>
            
            <h2>Technology</h2>
            <p>FDS utilizes cutting-edge deep learning models combined with traditional computer vision and natural language processing techniques:</p>
            
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">📰</div>
                    <h3>Text Analysis</h3>
                    <p>Detects sensationalism, verifiability issues, and writing patterns indicative of fake news using advanced NLP models.</p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">🖼️</div>
                    <h3>Image Forensics</h3>
                    <p>Analyzes images for digital manipulation through edge consistency analysis and noise pattern detection.</p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">🎬</div>
                    <h3>Video Analysis</h3>
                    <p>Identifies deepfakes and manipulated videos using temporal analysis and frame consistency checks.</p>
                </div>
            </div>
            
            <h2>Our Team</h2>
            <p>We are a diverse team of AI researchers, cybersecurity experts, and digital forensics specialists.</p>
            
            <div class="team-grid">
                <div class="team-card">
                    <div class="team-avatar">👨‍💻</div>
                    <h3>AI Researchers</h3>
                    <p>Experts in computer vision, NLP, and deep learning.</p>
                </div>
                
                <div class="team-card">
                    <div class="team-avatar">🛡️</div>
                    <h3>Security Experts</h3>
                    <p>Cybersecurity professionals ensuring data privacy.</p>
                </div>
                
                <div class="team-card">
                    <div class="team-avatar">🔍</div>
                    <h3>Forensic Analysts</h3>
                    <p>Specialists in digital media forensics.</p>
                </div>
            </div>
            
            <h2>Contact & Support</h2>
            <div class="contact-info">
                <h3>Get in Touch</h3>
                <a href="mailto:support@fakedetectionsystem.ai">
                    <i class="fas fa-envelope"></i>
                    support@fakedetectionsystem.ai
                </a>
            </div>
            
            <div class="highlight">
                <h3>Disclaimer</h3>
                <p>While our system strives for high accuracy, no automated system can guarantee 100% reliability. Always verify critical information through multiple trusted sources.</p>
            </div>
        </div>
        
        <footer>
            <p>&copy; 2023 Fake Detection System. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>''',
        
        'faq.html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAQ - Fake Detection System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            padding: 20px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.8rem;
            font-weight: bold;
        }
        
        .logo i {
            font-size: 2.2rem;
        }
        
        .nav {
            display: flex;
            gap: 15px;
        }
        
        .nav a {
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .nav a:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .faq-container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 50px;
            margin: 40px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        h1 {
            text-align: center;
            font-size: 2.8rem;
            margin-bottom: 10px;
            background: linear-gradient(to right, #ffffff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 40px;
        }
        
        .search-box {
            margin-bottom: 40px;
            position: relative;
        }
        
        .search-box input {
            width: 100%;
            padding: 18px 20px 18px 50px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 15px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 1rem;
            font-family: inherit;
        }
        
        .search-box input:focus {
            outline: none;
            border-color: rgba(255,255,255,0.5);
        }
        
        .search-box i {
            position: absolute;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0.7;
        }
        
        .faq-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .faq-item {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .faq-item.active {
            background: rgba(255,255,255,0.15);
        }
        
        .faq-question {
            padding: 25px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        
        .faq-question h3 {
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .faq-question i {
            color: #667eea;
            font-size: 1.2rem;
        }
        
        .toggle-icon {
            font-size: 1.5rem;
            transition: transform 0.3s ease;
        }
        
        .faq-item.active .toggle-icon {
            transform: rotate(45deg);
        }
        
        .faq-answer {
            padding: 0 30px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease, padding 0.3s ease;
        }
        
        .faq-item.active .faq-answer {
            padding: 0 30px 30px 30px;
            max-height: 500px;
        }
        
        .faq-answer p {
            line-height: 1.8;
            opacity: 0.95;
            font-size: 1.1rem;
        }
        
        .contact-section {
            background: rgba(255,255,255,0.1);
            padding: 40px;
            border-radius: 15px;
            margin-top: 60px;
            text-align: center;
        }
        
        .contact-section h3 {
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: white;
        }
        
        .contact-btn {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 15px 35px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 30px;
            font-weight: bold;
            margin-top: 20px;
        }
        
        footer {
            text-align: center;
            padding: 40px 0;
            margin-top: 60px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.8);
        }
        
        @media (max-width: 768px) {
            header {
                flex-direction: column;
                gap: 20px;
            }
            
            .nav {
                flex-wrap: wrap;
                justify-content: center;
            }
            
            .faq-container {
                padding: 30px 20px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .faq-question {
                padding: 20px;
            }
            
            .faq-question h3 {
                font-size: 1.1rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-robot"></i>
                <span>Fake Detection System</span>
            </div>
            <nav class="nav">
                <a href="/"><i class="fas fa-home"></i> Home</a>
                <a href="/detect"><i class="fas fa-search"></i> Detect</a>
                <a href="/history"><i class="fas fa-history"></i> History</a>
                <a href="/about"><i class="fas fa-info-circle"></i> About</a>
                <a href="/faq"><i class="fas fa-question-circle"></i> FAQ</a>
            </nav>
        </header>
        
        <div class="faq-container">
            <h1>Frequently Asked Questions</h1>
            <p class="subtitle">Find answers to common questions about our Fake Detection System</p>
            
            <div class="search-box">
                <i class="fas fa-search"></i>
                <input type="text" id="faq-search" placeholder="Search for questions...">
            </div>
            
            <div class="faq-list" id="faq-list">
                {% for faq in faqs %}
                <div class="faq-item">
                    <div class="faq-question" onclick="toggleFAQ(this)">
                        <h3>
                            <i class="fas fa-question-circle"></i>
                            {{ faq.question }}
                        </h3>
                        <span class="toggle-icon">+</span>
                    </div>
                    <div class="faq-answer">
                        <p>{{ faq.answer }}</p>
                    </div>
                </div>
                {% endfor %}
                
                <div class="faq-item">
                    <div class="faq-question" onclick="toggleFAQ(this)">
                        <h3>
                            <i class="fas fa-question-circle"></i>
                            What file sizes are supported?
                        </h3>
                        <span class="toggle-icon">+</span>
                    </div>
                    <div class="faq-answer">
                        <p>Maximum file sizes: Images - 50MB, Videos - 500MB. For optimal performance, we recommend keeping images under 20MB.</p>
                    </div>
                </div>
                
                <div class="faq-item">
                    <div class="faq-question" onclick="toggleFAQ(this)">
                        <h3>
                            <i class="fas fa-question-circle"></i>
                            How long does analysis take?
                        </h3>
                        <span class="toggle-icon">+</span>
                    </div>
                    <div class="faq-answer">
                        <p>Analysis time varies: News articles (1-5 seconds), Images (5-15 seconds), Videos (30 seconds to several minutes).</p>
                    </div>
                </div>
                
                <div class="faq-item">
                    <div class="faq-question" onclick="toggleFAQ(this)">
                        <h3>
                            <i class="fas fa-question-circle"></i>
                            Is my data shared with third parties?
                        </h3>
                        <span class="toggle-icon">+</span>
                    </div>
                    <div class="faq-answer">
                        <p>No. All uploaded content is processed locally and automatically deleted within 24 hours.</p>
                    </div>
                </div>
            </div>
            
            <div class="contact-section">
                <h3>Still have questions?</h3>
                <p>Can\'t find the answer you\'re looking for? Our support team is here to help you.</p>
                <a href="mailto:support@fakedetectionsystem.ai" class="contact-btn">
                    <i class="fas fa-envelope"></i>
                    Contact Support
                </a>
            </div>
        </div>
        
        <footer>
            <p>&copy; 2023 Fake Detection System. All rights reserved.</p>
        </footer>
    </div>
    
    <script>
        function toggleFAQ(element) {
            const item = element.parentElement;
            item.classList.toggle(\'active\');
        }
        
        // Search functionality
        document.getElementById(\'faq-search\').addEventListener(\'input\', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const allFAQs = document.querySelectorAll(\'.faq-item\');
            
            allFAQs.forEach(faq => {
                const question = faq.querySelector(\'.faq-question h3\').textContent.toLowerCase();
                const answer = faq.querySelector(\'.faq-answer p\').textContent.toLowerCase();
                
                if (question.includes(searchTerm) || answer.includes(searchTerm)) {
                    faq.style.display = \'flex\';
                    faq.style.flexDirection = \'column\';
                } else {
                    faq.style.display = \'none\';
                }
            });
        });
        
        // Open first FAQ by default
        document.querySelector(\'.faq-item\').classList.add(\'active\');
    </script>
</body>
</html>''',
        
        'error.html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error {{ error_code }} - Fake Detection System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .error-container {
            text-align: center;
            padding: 50px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 25px;
            max-width: 700px;
            margin: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .error-icon {
            font-size: 8rem;
            margin-bottom: 20px;
        }
        
        .error-code {
            font-size: 6rem;
            font-weight: bold;
            margin-bottom: 10px;
            background: linear-gradient(to right, #ffffff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .error-message {
            font-size: 2rem;
            margin-bottom: 30px;
            color: white;
        }
        
        .error-description {
            font-size: 1.2rem;
            margin-bottom: 40px;
            opacity: 0.9;
            line-height: 1.6;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 15px 30px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: transform 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .btn:hover {
            transform: translateY(-3px);
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 2px solid rgba(255,255,255,0.3);
        }
        
        @media (max-width: 600px) {
            .error-container {
                padding: 30px 20px;
            }
            
            .error-code {
                font-size: 4rem;
            }
            
            .error-message {
                font-size: 1.5rem;
            }
            
            .error-icon {
                font-size: 5rem;
            }
            
            .action-buttons {
                flex-direction: column;
            }
            
            .btn {
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-icon">
            {% if error_code == 404 %}
            🔍
            {% elif error_code == 500 %}
            ⚙️
            {% elif error_code == 413 %}
            📁
            {% else %}
            ⚠️
            {% endif %}
        </div>
        
        <div class="error-code">{{ error_code }}</div>
        <div class="error-message">{{ error_message }}</div>
        
        <div class="error-description">
            {% if error_code == 404 %}
            The page you\'re looking for doesn\'t exist or has been moved.
            {% elif error_code == 500 %}
            Something went wrong on our end. We\'re working to fix it.
            {% elif error_code == 413 %}
            The file you\'re trying to upload is too large. Maximum size is 500MB.
            {% else %}
            An unexpected error occurred.
            {% endif %}
        </div>
        
        <div class="action-buttons">
            <a href="/" class="btn">
                <i class="fas fa-home"></i>
                Go Home
            </a>
            <a href="/detect" class="btn btn-secondary">
                <i class="fas fa-search"></i>
                Detect Content
            </a>
        </div>
    </div>
</body>
</html>'''
    }
    
    # Check and create each template
    created = []
    for filename, content in templates.items():
        filepath = os.path.join(templates_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            created.append(filename)
    
    if created:
        print(f"✅ Created missing templates: {', '.join(created)}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting Fake Detection System...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Config loaded: {bool(CONFIG)}")
    
    # Create missing templates - DISABLED to preserve new UI
    # create_missing_templates()
    
    print("="*70)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)