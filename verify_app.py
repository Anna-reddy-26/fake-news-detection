
import unittest
import os
import sys

# Ensure we can import app
sys.path.append(os.getcwd())

from app import app

class TestFakeDetectionApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_home_page(self):
        print("\nTesting Home Page...")
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fake Detection System', response.data)
        print("[OK] Home Page OK")

    def test_detect_page_get(self):
        print("\nTesting Detect Page (GET)...")
        response = self.client.get('/detect')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analyze', response.data)
        print("[OK] Detect Page (GET) OK")

    def test_detect_news_empty(self):
        print("\nTesting Detect News (Empty)...")
        response = self.client.post('/detect', data={
            'analysis_type': 'news',
            'text': ''
        }, follow_redirects=True)
        self.assertIn(b'provide sufficient text', response.data)
        print("[OK] Detect News (Empty) handled correctly")
    
    def test_detect_news_valid(self):
        print("\nTesting Detect News (Valid)...")
        try:
            response = self.client.post('/detect', data={
                'analysis_type': 'news',
                'text': 'This is a test news article to verify the prediction endpoint works.'
            })
            self.assertEqual(response.status_code, 200)
            print("[OK] Detect News (Valid) OK")
        except Exception as e:
            print(f"[FAIL] Detect News (Valid) Failed: {e}")
            raise

if __name__ == '__main__':
    unittest.main()
