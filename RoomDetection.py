# RoomDetection - Implementation of bag of visual words using sift features and training a support vector machine.
# This part was used for training the classification model of the rooms.

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


# Each folder name inside the training data will serve as the class name
training_set_path = 'Training_data'
categories = os.listdir(training_set_path)

# match each picture in training data with its label
images_paths = []
images_classes = []
class_id = 0


for category in categories:
    dir = os.path.join(training_set_path, category)
    category_path = [os.path.join(dir, f) for f in os.listdir(dir)]
    images_paths += category_path
    images_classes += [class_id] * len(category_path)
    class_id += 1

# Feature extraction: Get descriptors and store them inside a list
n_features = 50
desc_list = []
sift = cv2.SIFT_create()
img_size = 400
for image_path in images_paths:
    im0 = cv2.imread(image_path)
    im = cv2.resize(im0, (img_size, img_size))
    kpts, des1 = sift.detectAndCompute(im, None)
    desc_list.append((image_path, des1))

# Create numpy array for feature vectors.
descriptors = desc_list[0][1]
for image_path, desc in desc_list[1:]:
    descriptors = np.vstack((descriptors, desc))
descriptors_fp = descriptors.astype(float)

# k-means and create features histogram
k = 1500

kmeans = KMeans(n_clusters=k).fit(descriptors_fp)
voc = kmeans.cluster_centers_

plt.scatter(descriptors_fp[:, 0], descriptors_fp[:, 1])
plt.scatter(voc[:, 0], voc[:, 1], c='r')
plt.show()

# calculate the features' histogram and represent them as a feature vector
im_features = np.zeros((len(images_paths), k), "float32")
for i in range(len(images_paths)):
    words, distance = vq(desc_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1


# Standard scaler was used to remove the mean from the dataset (if exists)

std_scaler = StandardScaler().fit(im_features)
im_features = std_scaler.transform(im_features)

# clf = LinearSVC(max_iter=10000)
svc_linear_clf = SVC(kernel='linear', probability=True)
svc_poly_clf = SVC(kernel='poly', degree=4, probability=True)
svc_rbf_clf = SVC(kernel='rbf', probability=True)
linear_svc_clf = LinearSVC(max_iter=10000)
clf_dict = {svc_rbf_clf: 'RBF SVC', svc_poly_clf: 'Polynomial SVC', svc_linear_clf: 'Linear kernel SVC', linear_svc_clf: 'LinearSVC'}
clfs = clf_dict.keys()

num_folds = 5
k_fold = KFold(n_splits=num_folds, random_state=None)

avg_score = []
data_frame = np.column_stack((im_features,np.array(images_classes)))
np.random.shuffle(data_frame)
X = data_frame[:, :-1]
y = data_frame[:, -1:].ravel()
fitted_clfs = []
for clf in clfs:
    acc_score = []
    print(clf_dict[clf])
    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        print(classification_report(y_test, y_pred))
        acc_score.append(acc)
        fitted_clfs.append(clf)

    avg_score.append(sum(acc_score)/num_folds)
    print("Accuracy of each fold - {}".format(acc_score))

best_clf_index = avg_score.index(max(avg_score))
print("Avg accuracy Score: {}".format(avg_score))
best_clf = fitted_clfs[best_clf_index]
print("The best svm kernel is: {}".format(best_clf_index))
print(best_clf_index)
# Re-fit best classifier based on results from k-fold
best_clf.fit(X, y)
# Save the SVM
joblib.dump((best_clf, categories, std_scaler, k, voc), "Try2.pkl", compress=3)
