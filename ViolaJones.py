import os
import cv2
import face_recognition
from PIL import Image
import Camera
import AudioOutput as Audio
import SpeechRecognition as Sr


# Face detection and recognition won't be able to find faces if the images are rotated, therefore should see in the plot
# if the image was correctly uploaded or retake the photo. (Using droid cam).

# After image is shown, exit it so the model will continue.

def face_detection():
    cap = cv2.VideoCapture(Camera.address)
    name = ""
    if cap.isOpened():
        ret, frame = cap.read()
        cv2.imwrite('tmp.jpg', frame)
        cv2.imshow('Face', frame)
        cv2.waitKey(0)
        if ret:
            Audio.speak("What is the person's name")
            name = Sr.Listen()
        cap.release()
        cv2.waitKey(2000)
    inc_size = 100
    image = face_recognition.load_image_file('tmp.jpg')
    cv2.imshow('one', image)
    cv2.waitKey(0)
    # os.remove('tmp.jpg')
    face_locations = face_recognition.face_locations(image)
    print(face_locations)
    if len(face_locations) != 1:
        return "null"
    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = image[top-inc_size:bottom+inc_size, left-inc_size:right+inc_size]
        pil_image = Image.fromarray(face_image)
        pil_image.show()
        pil_image.save(os.path.join('known_faces', name + '.jpg'))
    return "good"


def detect_faces():
    cap = cv2.VideoCapture(Camera.address)
    if cap.isOpened():
        ret, frame = cap.read()
        cv2.imwrite('tmp.jpg', frame)
        cv2.imshow('Face', frame)
        cv2.waitKey(0)
    cap.release()
    cv2.waitKey(2000)
    test_img = face_recognition.load_image_file('tmp.jpg')
    os.remove('tmp.jpg')
    path = 'known_faces'
    known_faces_encodings = []
    known_faces_names = []
    images_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for image_path in images_paths:
        current_face = face_recognition.load_image_file(image_path)
        known_face_encoding = face_recognition.face_encodings(current_face)[0]
        known_faces_encodings.append(known_face_encoding)
        known_faces_names.append(os.path.splitext(os.path.basename(image_path))[0])

    face_locations = face_recognition.face_locations(test_img)
    face_encodings = face_recognition.face_encodings(test_img, face_locations)

    names = []

    # Loop through faces in test image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)

        if True in matches:
            match_index = matches.index(True)
            name = known_faces_names[match_index]
            names.append(name)
    return names
