import cv2
import joblib
import numpy as np
import os
from scipy.cluster.vq import vq
from sklearn.metrics import accuracy_score, confusion_matrix


clf, classes_names, std_scaler, k, voc = joblib.load("Rooms_classifier.pkl")

test_path = 'Test_data'
testing_names = os.listdir(test_path)

# match each picture in training data with its label
images_paths = []
images_classes = []
class_id = 0

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
    images_paths += class_path
    images_classes += [class_id] * len(class_path)
    class_id += 1

desc_list = []
sift = cv2.SIFT_create()
img_size = 400
for image_path in images_paths:
    im0 = cv2.imread(image_path)
    im = cv2.resize(im0, (img_size, img_size))
    kpts, des = sift.detectAndCompute(im, None)
    desc_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array

descriptors = desc_list[0][1]
for image_path, descriptor in desc_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))


test_features = np.zeros((len(images_paths), k), "float32")
for i in range(len(images_paths)):
    words, distance = vq(desc_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

test_features = std_scaler.transform(test_features)

data_frame = np.column_stack((test_features, np.array(images_classes)))
X = data_frame[:, :-1]
y = data_frame[:, -1:].ravel()
y_pred = clf.predict(X)
acc = accuracy_score(y_pred, y)
print(acc)

# Print actual class names and predictions for each sample in order to perform comparisons.

actual_class = [classes_names[i] for i in images_classes]

predictions_ids = y_pred.tolist()
predictions_ids = [int(y) for y in predictions_ids]
predictions = [classes_names[i] for i in predictions_ids]

print("actual class =" + str(actual_class))
print("prediction =" + str(predictions))

accuracy = accuracy_score(actual_class, predictions)
confusion_matrix = confusion_matrix(actual_class, predictions)
print(confusion_matrix)
