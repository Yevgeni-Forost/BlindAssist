import cv2
import numpy as np
import os
import Camera as Cam
import joblib
from scipy.cluster.vq import vq

test_path = 'Test'


def load_images():
    sift = cv2.SIFT_create()
    des_list = []
    img_size = 400
    for image in os.listdir(test_path):
        img_path = os.path.join(test_path, image)
        im0 = cv2.imread(img_path)
        im = cv2.resize(im0, (img_size, img_size))
        kpts, des = sift.detectAndCompute(im, None)
        des_list.append((img_path, des))
    return des_list


def calculate_features(des_list, stdSlr, k, voc, num_img):
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))
    print(num_img)
    test_features = np.zeros((num_img, k), "float32")
    for i in range(num_img):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            test_features[i][w] += 1
    test_features = stdSlr.transform(test_features)
    return test_features


def most_frequent(names):
    return max(set(names), key=names.count)


def classify_rooms():
    Cam.capture_4_frames()
    clf, classes_names, stdSlr, k, voc = joblib.load("Rooms_classifier.pkl")
    data = load_images()
    num_img = len(data)
    test_features = calculate_features(data, stdSlr, k, voc, num_img)
    y_pred = clf.predict(test_features)

    # Report true class names so they can be compared with predicted classes
    # actual_class = [classes_names[i] for i in classes_names]
    # Perform the predictions and report predicted class names.
    predictions_ids = y_pred.tolist()
    predictions_ids = [int(y) for y in predictions_ids]
    predictions = [classes_names[i] for i in predictions_ids]
    room = most_frequent(predictions)
    Cam.clr_folder()
    return room
