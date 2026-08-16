import firebase_admin
from firebase_admin import credentials, firestore

try:
    # Initialize Firebase
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

    # Initialize Firestore client
    db = firestore.client()

    # Write a test document
    users_ref = db.collection('users')
    doc_ref = users_ref.add({
        'name': 'Nikitha',
        'email': 'nikitha@example.com'
    })
    print("Document added with ID:", doc_ref[1].id)

    # Read all documents
    print("\nAll documents in 'users' collection:")
    docs = users_ref.stream()
    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')

except Exception as e:
    print("Error:", e)
