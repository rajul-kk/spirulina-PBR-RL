import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import random  # simulate sensor values

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Simulate sensor data
def get_sensor_data():
    # In practice, replace this with actual readings from simulation.py or real sensors
    return {
        "sensor_id": f"sensor_{random.randint(1,3)}",
        "temperature": round(random.uniform(20.0, 30.0), 2),
        "value": round(random.uniform(0.0, 100.0), 2),
        "timestamp": datetime.now().isoformat()
    }

# Push data to Firestore
data = get_sensor_data()
doc_ref = db.collection("sensor_readings").add(data)
print(f"Data added: {doc_ref[1].id}")
print(data)
