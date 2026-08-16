import firebase_admin
from firebase_admin import credentials, firestore
import time

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")  # make sure your key is correct
firebase_admin.initialize_app(cred)
db = firestore.client()

def on_snapshot(col_snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'ADDED':
            print(f"New data: {change.document.id} => {change.document.to_dict()}")

# Watch the collection
col_query = db.collection('sensor_readings')
query_watch = col_query.on_snapshot(on_snapshot)

print("Subscriber is running. Waiting for new sensor data...")

# Keep the script alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Subscriber stopped.")
