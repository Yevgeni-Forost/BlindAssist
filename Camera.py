import cv2
import os
import shutil


# Droid cam address - We used DroidCam application (Android) in order to perform the various testings in the project.
# We didn't test any other IP apps but DroidCam is really nice and allows to connect the phones camera via IP address.
# Both devices - Phone and the computer need to be on the same network.
# Change the ip and the port (without changing the mjpegfeed) to the ip and port that appears IP Cam Access.

address = 'http://192.168.0.137:4747/mjpegfeed'

# Sliding window for binary object classifier, ended up not being used due to poor performance of the classifiers.
# Will continue to work on that (Either training binary classifiers / deep learning model


def sliding_window(image):
    image = cv2.resize(image, (1200, 1200))
    step_size = 50
    (w_width, w_height) = (100, 100)  # window size
    count = 0
    path = "Test"
    for x in range(0, image.shape[1], step_size):
        for y in range(0, image.shape[0], step_size):
            window = image[x:x + w_width, y:y + w_height, :]
            if window is None:
                return
            cv2.imwrite(os.path.join(path, "frame{}.jpg".format(count)), window)
            count = count + 1


# Image pyramid for Binary object classification (Ended up not using).
# Image pyramid was originally planned to be used in the

# def image_pyramid(img):
#     min_size = 800
#     update = 100
#     h, w, c = img.shape
#     if h - update < min_size:
#         return img, 'null'
#     resized = cv2.resize(img, (h-update, w-update))
#     sliding_window(resized)
#     return resized, 'continue'


# Capturing 4 frames for room detection.
def capture_4_frames():
    cap = cv2.VideoCapture(address)
    path = 'Test'
    name = 1
    for i in range(4):
        ret, image = cap.read()
        cv2.imwrite(os.path.join(path, 'frame' + str(name) + '.jpg'), image)
        name = name + 1
        cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()


# Capturing gray image for bruteforce matcher algorithm and items detection.
def capture_gray():
    cap = cv2.VideoCapture(address)
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(2000)
    cap.release()
    return img_gray


# Clear Folder is used to clear the images that are taken and saved in each task (Sliding window / capturing images
# of the rooms etc). We are doing that as The Test folder is being used by us for all sorts of actions except BFMatcher.

def clr_folder():
    folder = 'Test'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# Hard negative mining - we combined the sliding window function with the classify function from ClassifyObject.py
# in order to apply improvements to the object classifier (which we unfortunately ended up not using due to
# poor performance). Further elaboration (commented) available in the ClassifyObject.py file.

# def hard_negative_mining():
#     path = "Test"
#     path_true = "True"
#     path_false = "False"
#
#     # Read image, perform resize and define window size (100,100) and a smaller step size to cover all options.
#
#     image = cv2.imread("test_img.jpg")  # your image path
#     image = cv2.resize(image, (1200,1200))
#     step_size = 50
#     (w_width, w_height) = (100, 100)  # window size
#     count = 0
#     for x in range(0, image.shape[0], step_size):
#         for y in range(0, image.shape[1], step_size):
#             window = image[x:x + w_width, y:y + w_height, :]
#             if window is not None:
#                 cv2.imwrite(os.path.join(path, "frame{}.jpg".format(count)), window)
#             count = count + 1
#             if Co.classify(2):
#                 cv2.imwrite(os.path.join(path_true, "frame{}.jpg".format(count)), window)
#             else:
#                 cv2.imwrite(os.path.join(path_false, "frame{}.jpg".format(count)), window)
