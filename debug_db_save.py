import db
import numpy as np
from flask import Flask

# Setup mock app
app = Flask(__name__)
app.root_path = '.'

# Initialize DB (will likely go to Mock if key missing, but we want to test the save function logic)
# To test Real DB serialization, we need the key. If user hasn't provided it, we can't fully repro "real" db failure 
# but we can see if the SDK *would* fail or if our logic throws.
# However, if it's in Mock mode, it just prints. 
# The user says "database is not working", implying they expect it to work.
# I will assume they added the key.

print("Initializing DB...")
db.init_db(app)

print("Attempting to save data with Numpy types...")
try:
    # Simulate data from ImageDetector
    data = {
        'analysis_id': 'DEBUG_TEST_001',
        'prediction': 'Fake',
        'confidence': np.float32(0.95),  # Numpy type
        'fake_probability': np.float64(0.95123), # Numpy type
        'media_type': 'image',
        'risk_level': 'High Risk',
        'analysis_details': {
            'edge_density': np.float32(0.5)
        }
    }
    
    # Force mock_mode to False to test the "Real" saving logic branch if we can,
    # but if db client is None, it won't do anything.
    # We can try to use the logic in db.py.
    
    # If we are in mock mode (likely here), we won't see the Firestore error.
    # But I strongly suspect `convert_to_serializable` is needed.
    
    db.save_analysis(data)
    print("Save called.")

except Exception as e:
    print(f"Caught expected error: {e}")
