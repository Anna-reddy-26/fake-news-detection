
import unittest
import os
import sys
import numpy as np
import cv2
import io

# Ensure we can import app
sys.path.append(os.getcwd())

from app import app

class TestMediaDetection(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        
        # Create dummy image
        self.img_path = 'dummy_test.jpg'
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (255, 0, 0) # Blue image
        cv2.imwrite(self.img_path, img)
        
        # Create dummy video
        self.vid_path = 'dummy_test.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.vid_path, fourcc, 30.0, (112, 112))
        for _ in range(30):
            frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)
        if os.path.exists(self.vid_path):
            os.remove(self.vid_path)
            
    def test_detect_image(self):
        print("\nTesting Detect Image...")
        with open(self.img_path, 'rb') as f:
            data = {
                'analysis_type': 'image',
                'file': (f, 'dummy_test.jpg')
            }
            response = self.client.post('/detect', data=data, content_type='multipart/form-data', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Analysis Result', response.data)
            print("[OK] Detect Image OK")

    def test_detect_video(self):
        print("\nTesting Detect Video...")
        with open(self.vid_path, 'rb') as f:
            data = {
                'analysis_type': 'video',
                'file': (f, 'dummy_test.mp4')
            }
            # Note: Video analysis might take time or fail if model expects specific format
            # But the code should handle it.
            response = self.client.post('/detect', data=data, content_type='multipart/form-data', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Analysis Result', response.data)
            print("[OK] Detect Video OK")

if __name__ == '__main__':
    unittest.main()
