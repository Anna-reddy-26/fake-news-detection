import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Configuration
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'fake_detection_db')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 524288000))
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    
    # Model Paths
    IMAGE_MODEL_PATH = os.getenv('IMAGE_MODEL_PATH', 'models/image_model.pth')
    VIDEO_MODEL_PATH = os.getenv('VIDEO_MODEL_PATH', 'models/video_model.pth')
    NEWS_MODEL_PATH = os.getenv('NEWS_MODEL_PATH', 'models/news_model.pkl')
    
    # Dataset Paths
    DATASET_PATH = os.getenv('DATASET_PATH', 'Dataset')
    MANIPULATED_VIDEOS_PATH = os.getenv('MANIPULATED_VIDEOS_PATH', 'Dataset/DFD_manipulated_sequences')
    ORIGINAL_VIDEOS_PATH = os.getenv('ORIGINAL_VIDEOS_PATH', 'Dataset/DFD_original_sequences')
    IMAGES_PATH = os.getenv('IMAGES_PATH', 'Dataset/Images')
    NEWS_FAKE_PATH = os.getenv('NEWS_FAKE_PATH', 'Dataset/news/Fake.csv')
    NEWS_TRUE_PATH = os.getenv('NEWS_TRUE_PATH', 'Dataset/news/True.csv')
    
    # Training Parameters
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    EPOCHS = int(os.getenv('EPOCHS', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    IMAGE_SIZE = tuple(map(int, os.getenv('IMAGE_SIZE', '224,224').split(',')))
    
    # Feature Extraction Parameters
    MAX_FEATURES = int(os.getenv('MAX_FEATURES', 1000))
    MAX_SEQUENCE_LENGTH = int(os.getenv('MAX_SEQUENCE_LENGTH', 100))
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')