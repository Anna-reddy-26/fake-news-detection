import unittest
import os
import sys
from flask import url_for

# Ensure we can import app
sys.path.append(os.getcwd())

from app import app, db

class TestAuthFlow(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.secret_key = 'test-secret'
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        # Use a random email for each test run to avoid collision if DB is real
        import random
        self.test_email = f"test_{random.randint(1000, 9999)}@example.com"
        self.test_password = "password123"
        self.test_name = "Test User"

    def tearDown(self):
        self.app_context.pop()

    def test_auth_flow(self):
        print(f"\nTesting Auth Flow with email: {self.test_email}")
        
        # 1. Access protected route -> should redirect
        print("1. Accessing /detect without login...")
        response = self.client.get('/detect', follow_redirects=True)
        # Should be redirected to login
        self.assertIn(b'Login', response.data)
        print("[OK] Redirected to login")

        # 2. Register
        print("2. Registering new user...")
        response = self.client.post('/register', data={
            'name': self.test_name,
            'email': self.test_email,
            'password': self.test_password
        }, follow_redirects=True)
        self.assertIn(b'Login', response.data) # Should redirect to login page
        print("[OK] Registration successful")

        # 3. Login
        print("3. Logging in...")
        response = self.client.post('/login', data={
            'email': self.test_email,
            'password': self.test_password
        }, follow_redirects=True)
        self.assertIn(b'Dashboard', response.data) # Should redirect to index/dashboard
        self.assertIn(self.test_name.encode(), response.data) # Name should be in nav
        print("[OK] Login successful")

        # 4. Access protected route -> should succeed
        print("4. Accessing /detect after login...")
        response = self.client.get('/detect')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analysis', response.data)
        print("[OK] Access granted")

        # 5. Logout
        print("5. Logging out...")
        response = self.client.get('/logout', follow_redirects=True)
        self.assertIn(b'Login', response.data) # Should see login link in nav
        print("[OK] Logout successful")

        # 6. Access protected route -> should redirect again
        print("6. Accessing /detect after logout...")
        response = self.client.get('/detect', follow_redirects=True)
        self.assertIn(b'Login', response.data)
        print("[OK] Access denied")

if __name__ == '__main__':
    unittest.main()
