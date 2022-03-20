import cv2
import Camera as Cam
import os
from skimage.feature import orb

orb = cv2.ORB_create()


def calc_desc(images):
    desc_list = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desc_list.append(des)
    return desc_list


def get_match(img, desc_list):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    match_list = []
    best_match = -1
    try:
        for des in desc_list:
            matches = bf.knnMatch(des, des2, k=2)
            correct_match = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    correct_match.append([m])
            match_list.append(len(correct_match))
    except:
        pass
    if len(match_list) > 0:
        if max(match_list) > 14:
            best_match = match_list.index(max(match_list))
    return best_match


def capture_image(desc_list, class_names, img_gray):
    result = ""
    match_id = get_match(img_gray, desc_list)
    if id != -1:
        result = str(class_names[match_id])
        print("match is : " + result)
    return result


def bf_matcher():
    img_gray = Cam.capture_gray()
    cv2.imshow('img', img_gray)
    cv2.waitKey(0)
    path = 'BfImages'
    images_names = os.listdir(path)
    images = []
    class_names = []

    for name in images_names:
        curr_img = cv2.imread(f'{path}/{name}', 0)
        images.append(curr_img)
        class_names.append(os.path.splitext(name)[0])
    print(class_names)
    desc_list = calc_desc(images)
    item = capture_image(desc_list, class_names, img_gray)
    return item
