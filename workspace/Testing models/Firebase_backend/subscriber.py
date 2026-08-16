import json
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    data = json.loads(message.payload.decode())
    print(" Data received by ML subscriber:", data)

client = mqtt.Client()
client.connect("test.mosquitto.org")
client.subscribe("validated/sensor_data")
client.on_message = on_message

print(" Listening for data...")
client.loop_forever()
