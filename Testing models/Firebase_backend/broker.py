import json
import paho.mqtt.publish as publish

def receive_sensor_data():
    # Example data from sensor
    data = {
        "sensor_id": "temp_1",
        "value": 28.6,
        "timestamp": "2025-11-05T12:00:00Z"
    }

    # Validate data
    if 0 <= data["value"] <= 60:
        print(" Valid data received:", data)
        # Send to subscribers
        publish.single("validated/sensor_data", json.dumps(data), hostname="test.mosquitto.org")
    else:
        print(" Invalid data. Ignored.")

receive_sensor_data()
