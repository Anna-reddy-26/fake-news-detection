import os
import json

key_path = 'serviceAccountKey.json'
abs_path = os.path.abspath(key_path)

print(f"Checking for key at: {abs_path}")

if os.path.exists(key_path):
    print("[OK] File exists!")
    try:
        with open(key_path, 'r') as f:
            data = json.load(f)
            print("[OK] File is valid JSON.")
            print(f"Project ID: {data.get('project_id', 'Unknown')}")
    except Exception as e:
        print(f"[ERR] File exists but is invalid: {e}")
else:
    print("[ERR] File NOT found.")
    print("Contents of current directory:")
    for f in os.listdir('.'):
        print(f" - {f}")
