# This part ended up not being used. Originally, the plan was to train binary classifiers for different objects.
# The problem we encountered was false-positive / false-negative classifications for the object.
# We chose to deal with the issue using hard negative mining - after training a basic model we started taking
# more photos for test purposes - splitting them into windows (hard_negative_mining function is in camera class).
# The images that were classified badly were inserted to their true folder/class and we re-trained the model.
# The problem was that the training data quickly became unbalanced.
# The results were that the objects weren't recognized at all when testing.
# We then tried over-sampling the minority class which caused over-fitting (results were around 95% accuracy in training
# but there was no improvement on the testing results.
# We also tried removing the original No object class training photos to see how the classifier will deal with the
# more difficult samples that were found when we performed the hard negative mining and accuracy dropped down to 60%.


# Training the binary classifiers code:

# import cv2
# import joblib
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler
# from skimage import color
# from skimage.feature import hog
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import KFold
#
#
# # Each folder name inside the training data will serve as the class name
# training_set_path = 'wallet'
# categories = os.listdir(training_set_path)
#
# images_paths = []
# images_classes = []
# class_id = 0
# # Load categories (labels) for each image in the training data path, assign class ids for classes
# for category in categories:
#     dir = os.path.join(training_set_path, category)
#     category_path = [os.path.join(dir, f) for f in os.listdir(dir)]
#     images_paths += category_path
#     images_classes += [class_id] * len(category_path)
#     class_id += 1
#
#
# # Load images, resize and convert to grayscale for feature extraction
# def load_data(images_paths):
#     images_list = []
#     img_size = 100
#     for image_path in images_paths:
#         im0 = cv2.imread(image_path)
#         im = cv2.resize(im0, (img_size, img_size))
#         images_list.append(im)
#     data = np.array(images_list)
#     data_gray = [color.rgb2gray(i) for i in data]
#     return data_gray
#
#
# def shuffle_oversampled(X, y):
#     random_state = np.random.get_state()
#     np.random.shuffle(X)
#     np.random.set_state(random_state)
#     np.random.shuffle(y)
#     return X, y
#
# # Feature extraction using hog features, creating lists of features and image representation
# data = load_data(images_paths)
# ppc = 12
# hog_images = []
# hog_features = []
# for image in data:
#     fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2), block_norm='L2',
#                         visualize=True)
#     hog_images.append(hog_image)
#     hog_features.append(fd)
#
# #for i in range(15,30):
# plt.imshow(hog_images[9])
# plt.show()
#
# clf = svm.SVC(kernel='poly', degree=3, probability=True)
# # clf = RandomForestClassifier(n_estimators=200, random_state=30)
# # clf = LogisticRegression()
#
#
# # Create a shuffled data frame containing the features and labels of the data, use k-fold to estimate accuracy.
# labels = np.reshape(images_classes, (len(images_classes), 1))
# hog_features = np.array(hog_features, "float32")
# data_frame = np.hstack((hog_features, labels))
# X = data_frame[:, :-1]
# y = data_frame[:, -1:].ravel()
# oversample = RandomOverSampler(sampling_strategy='minority')
# # X, y = oversample.fit_resample(X_temp, y_temp)
# X, y = shuffle_oversampled(X, y)
# k = 5
# k_fold = KFold(n_splits=5, random_state=None)
# acc_score = []
#
# for train_index, test_index in k_fold.split(X):
#     cm = []
#     X_train, X_test = X[train_index, :], X[test_index, :]
#     y_train, y_test = y[train_index], y[test_index]
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_pred, y_test)
#     print(classification_report(y_test, y_pred))
#     acc_score.append(acc)
#     predictions_probability = [clf.predict_proba(X_test)]
#     ## Check classes
#     # Report true class names so they can be compared with predicted classes
#     class_ids = y_test.tolist()
#     actual_class_ids = [int(y) for y in class_ids]
#     actual_class = [categories[i] for i in actual_class_ids]
#     # Perform the predictions and report predicted class names.
#     predictions_ids = y_pred.tolist()
#     predictions_ids = [int(y) for y in predictions_ids]
#     predictions = [categories[i] for i in predictions_ids]
#     cm = confusion_matrix(actual_class, predictions)
#     print(cm)
#     # Print the true class and Predictions
#     print("actual class =" + str(actual_class))
#     print("prediction =" + str(predictions))
#     print(predictions_probability)
#
# avg_score = sum(acc_score)/k
#
# print("Accuracy of each fold - {}".format(acc_score))
# print("Avg accuracy Score: {}".format(avg_score))
#
# clf.fit(X, y)
# joblib.dump((clf, categories), "wallet_classifier.pkl")

# -------------------------------------------------------------------------------------------------------------


# Performing the classification code: (Using Sliding window / Image pyramid)


# ClassifyObject - Loading the needed model upon request, performing feature extraction and classification

# import os
# import numpy as np
# import joblib
# import cv2
# import Camera as Cam
# from skimage import color
# from skimage.feature import hog
#
# test_path = 'Test'
#
# def load_model(index):
#     models = ['Hat', 'Key', 'Wallet']
#     clf, categories = joblib.load(models[index]+"_classifier.pkl")
#     return clf, categories
#
#
# def load_images():
#     images_list = []
#     img_size = 100
#     for image in os.listdir(test_path):
#         img_path = os.path.join(test_path, image)
#         im0 = cv2.imread(img_path)
#         im = cv2.resize(im0, (img_size, img_size))
#         images_list.append(im)
#     data = np.array(images_list)
#     data_gray = [color.rgb2gray(i) for i in data]
#     return data_gray
#
#
# Extract hog features and return numpy array for classification
# def extract_features(data):
#     ppc = 12
#     hog_images = []
#     hog_features = []
#     for image in data:
#         fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2),
#         block_norm='L2', visualize=True)
#         hog_images.append(hog_image)
#         hog_features.append(fd)
#     hog_features = np.array(hog_features, "float32")
#     return hog_features
#
#
# def classify(index):
#     not_found = True
#     # img = Cam.capture_object()
#     clf, categories = load_model(index)
#     object = categories[0]
#     if categories[0] == 'None':
#         object = 'Wallet'
#     # cur_img = img
#     while not_found:
#         images = load_images()
#         test_features = extract_features(images)
#         y_pred = clf.predict(test_features)
#         predictions_ids = y_pred.tolist()
#         predictions_ids = [int(y) for y in predictions_ids]
#         predictions = [categories[i] for i in predictions_ids]
#         print(predictions)
#         if object in predictions:
#             Cam.clr_folder()
#             return True
#         Cam.clr_folder()
#         return False
#         # Cam.clr_folder()
#         --- Call image pyramid - resize the image to obtain more information in a window.
#         # cur_img, report = Cam.image_pyramid(cur_img)
#         # if report == 'null':
#         #     return False
