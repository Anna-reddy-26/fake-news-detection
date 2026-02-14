import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime
import json
import random
import numpy as np

# Global DB client
db = None
mock_mode = False

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return convert_to_serializable(obj.tolist())
    return obj

def init_db(app):
    global db, mock_mode
    
    cred_path = os.path.join(app.root_path, 'serviceAccountKey.json')
    
    if os.path.exists(cred_path):
        try:
            cred = credentials.Certificate(cred_path)
            # Check if already initialized to avoid error on reload
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
                print("Firebase Admin Initialized.")
            else:
                print("Firebase Admin already initialized.")
            
            db = firestore.client()
            print("Firebase Firestore connected successfully!")
        except Exception as e:
            print(f"Firebase init error: {e}")
            print("Switching to MOCK DATABASE mode.")
            mock_mode = True
    else:
        print(f"serviceAccountKey.json not found at {cred_path}")
        print("Switching to MOCK DATABASE mode.")
        mock_mode = True

def save_analysis(data):
    """Save an analysis result to Firestore"""
    if mock_mode:
        print(f"[MOCK] Saving analysis: {data.get('analysis_id')}")
        return
    
    try:
        if db:
            # Clean data (convert numpy types)
            clean_data = convert_to_serializable(data)
            
            # Add server timestamp
            clean_data['created_at'] = firestore.SERVER_TIMESTAMP
            db.collection('analyses').add(clean_data)
            print(f"Saved analysis {clean_data.get('analysis_id')} to Firestore")
    except Exception as e:
        print(f"Failed to save analysis: {e}")

def get_history(limit=50):
    """Fetch recent analyses from Firestore"""
    if mock_mode:
        return get_mock_history()
    
    try:
        if db:
            docs = db.collection('analyses')\
                .order_by('created_at', direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            
            history = []
            for doc in docs:
                data = doc.to_dict()
                # Convert timestamp to string if needed, mostly handled by Jinja or string format in app
                history.append(data)
            return history
    except Exception as e:
        print(f"Failed to fetch history: {e}")
        return []

def get_stats():
    """Calculate dashboard stats"""
    if mock_mode:
        return get_mock_stats()
    
    try:
        if db:
            # Note: Counting all documents can be expensive/slow in Firestore.
            # For a small app, this is fine. For scale, use aggregation queries or counters.
            # Using aggregation query if available in this SDK version, else client side count (limit 1000)
            
            docs = db.collection('analyses').limit(1000).stream()
            total = 0
            fake_count = 0
            recent_activity = []
            
            for doc in docs:
                data = doc.to_dict()
                total += 1
                if data.get('prediction') == 'Fake':
                    fake_count += 1
                
                # Get first 5 for recent activity
                if len(recent_activity) < 5:
                    recent_activity.append({
                        'type': data.get('media_type', 'unknown'),
                        'result': data.get('prediction', 'Unknown'),
                        'time': data.get('timestamp', 'Just now') # Simplification
                    })
            
            accuracy = 92.5 # Placeholder or calculated from Feedback if implemented
            
            return {
                'total_analysis': total,
                'fake_detected': fake_count,
                'accuracy_rate': f"{accuracy}%",
                'recent_activity': recent_activity
            }
    except Exception as e:
        print(f"Failed to fetch stats: {e}")
        return get_mock_stats()

def create_user(user_data):
    """Create a new user in Firestore"""
    if mock_mode:
        print(f"[MOCK] Creating user: {user_data.get('email')}")
        return True
    
    try:
        if db:
            # Check if user already exists
            user_ref = db.collection('users').document(user_data['email'])
            if user_ref.get().exists:
                return False
            
            user_data['created_at'] = firestore.SERVER_TIMESTAMP
            user_ref.set(user_data)
            print(f"Created user {user_data.get('email')}")
            return True
    except Exception as e:
        print(f"Failed to create user: {e}")
        return False

def get_user(email):
    """Get user by email"""
    if mock_mode:
        return {'email': email, 'name': 'Mock User', 'password': 'mock_password_hash'}
    
    try:
        if db:
            doc = db.collection('users').document(email).get()
            if doc.exists:
                return doc.to_dict()
    except Exception as e:
        print(f"Failed to get user: {e}")
    return None

# --- Mock Helpers ---

def get_mock_history():
    return [
        {
            'id': 'MOCK_001',
            'type': 'news',
            'title': 'Breaking News: Mock Data Active',
            'result': 'Fake',
            'confidence': 0.92,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'risk_level': 'High Risk'
        },
        {
            'id': 'MOCK_002',
            'type': 'image',
            'title': 'Demo Image Analysis',
            'result': 'Real',
            'confidence': 0.88,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'risk_level': 'Low Risk'
        }
    ]

def get_mock_stats():
    return {
        'total_analysis': 1234, # Mock numbers
        'fake_detected': 150,
        'accuracy_rate': "94.2%",
        'recent_activity': [
            {'type': 'news', 'result': 'Fake', 'time': 'Mock Time'},
            {'type': 'image', 'result': 'Real', 'time': 'Mock Time'}
        ]
    }
