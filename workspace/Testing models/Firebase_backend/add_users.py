# add_users.py
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app
cred = credentials.Certificate("serviceAccountKey.json")  # Make sure this file is in the same folder
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# List of users to add
users = [
    {"name": "Nikitha", "email": "nikitha@example.com"},
    {"name": "Bhavatharni", "email": "bhavatharni@example.com"},
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"}
]

# Add users to Firestore
for user in users:
    doc_ref = db.collection("users").add(user)
    print(f"Added user '{user['name']}' with ID: {doc_ref[1].id}")

print("\nAll users added successfully!")
