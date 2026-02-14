import db
from flask import Flask

# Create a dummy app context
app = Flask(__name__)
# Mock root path
app.root_path = '.'

print("Testing DB Initialization...")
db.init_db(app)

print("\nTesting Get Stats...")
stats = db.get_stats()
print(f"Stats: {stats}")

print("\nTesting Save Analysis...")
dummy_data = {
    'analysis_id': 'TEST_001',
    'prediction': 'Real', 
    'media_type': 'text',
    'timestamp': '2023-01-01'
}
db.save_analysis(dummy_data)

print("\nTesting Get History...")
history = db.get_history()
print(f"History count: {len(history)}")
print("First item:", history[0] if history else "None")

print("\nDB Verification Complete")
