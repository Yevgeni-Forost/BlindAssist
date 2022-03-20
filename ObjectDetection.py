import cv2
import numpy as np
# import Camera as Cam


def detect_object(obj_name ,img):
    # load dnn - yolo weights and configurations
    dnn = cv2.dnn.readNet("yoloV3.weights", "yoloV3.cfg")
    classes_names = []
    with open("coco.names", "r") as f:
        classes_names = [line.strip() for line in f.readlines()]
    layer_names = dnn.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in dnn.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes_names), 3))

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    dnn.setInput(blob)
    outputs = dnn.forward(output_layers)

    class_ids = []
    confident_levels = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            # print(class_id)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Used to draw bounding box on image to validate correctness of detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confident_levels.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confident_levels, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            if obj_name in str(classes_names[class_ids[i]]):
                return True


