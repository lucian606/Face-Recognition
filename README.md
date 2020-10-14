# Face-Recognition

Implementation of a face recognition algorithm, made on a Raspberry Pi 4 using Python and opencv library.
The algorithm uses a defined data set in order to identify the users.

How to run the script:
  - Run the faceset_script.py to create a data set
  - Run the face_trainer to create a trainer which can identify the users
  - Modify the face_recognition.py file, add your name into the names list (the id of a picture will be the index of the name)
  - Run the face_recognition.py to detect the faces
