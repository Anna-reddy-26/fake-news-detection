
import os
import sys

# Ensure we can import app
sys.path.append(os.getcwd())

print("Attempting to import app module...")
try:
    from app import news_detector, image_detector, video_detector
    print("[OK] App module imported successfully.")
except Exception as e:
    print(f"[FAIL] Failed to import app module: {e}")
    sys.exit(1)

def verify_detector(name, detector):
    print(f"\nVerifying {name}...")
    if detector is None:
        print(f"[FAIL] {name} is None!")
        return False
    
    if detector.model is None:
        print(f"[FAIL] {name}.model is None! Loading failed (check app logs).")
        return False
    
    print(f"[OK] {name} loaded successfully.")
    print(f"   Model type: {type(detector.model)}")
    return True

success = True
success &= verify_detector("NewsDetector", news_detector)
success &= verify_detector("ImageDetector", image_detector)
success &= verify_detector("VideoDetector", video_detector)

if success:
    print("\n[OK] All models verified successfully!")
    sys.exit(0)
else:
    print("\n[FAIL] One or more models failed to verify.")
    sys.exit(1)
