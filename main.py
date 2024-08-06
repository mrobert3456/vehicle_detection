import sys
import cv2 as cv
import numpy as np
from utils.image_roi import get_image_roi

f = open('data/coco.names', 'rb')
labels = list(n.decode('UTF-8').replace('\n', ' ').strip() for n in f.readlines())

path_to_weights = 'data/yolov3.weights'
path_to_cfg = 'data/yolov3.cfg'

# Loading trained YOLO v3 weights and cfg files
network = cv.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)

# GPU
network.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
network.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Minimum probability
probability_minimum = 0.6
# bounding boxes threshold non-maximum suppression
threshold = 0.2

# get the detection layers of yolov3
# all YOLO v3 layers
layers_all = network.getLayerNames()
layers_names_output = [layers_all[i - 1] for i in network.getUnconnectedOutLayers()]


def process_video(video_path: str):
    capture = cv.VideoCapture(video_path)
    h, w = None, None
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            break

        if w is None or h is None:
            # Slicing two elements from tuple
            h, w = frame.shape[:2]

        image_roi = get_image_roi(frame)
        # cv.imshow('frame', image_roi)

        blob = cv.dnn.blobFromImage(image_roi, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Forward pass with blob
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)

        # array for detected bounding boxes, confidences and class number
        bounding_boxes = []
        confidences = []
        class_numbers = []
        # output layers after feed forward pass
        for result in output_from_network:

            # all detections from current output layer
            for detected_objects in result:
                # class probabilities
                scores = detected_objects[5:]
                # index of the most probable class
                class_current = np.argmax(scores)
                # Getting probability values for current class
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Getting top left corner coordinates of bounding box
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        # Implementing non-maximum suppression of given bounding boxes
        # this will get only the relevant bounding boxes (there might be more which crosses each other, and etc)
        results = cv.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                # Bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                cv.rectangle(frame, (x_min, y_min),
                             (x_min + box_width, y_min + box_height),
                             (0, 255, 0), 2)

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    argCount = len(sys.argv)
    inputFile = str(sys.argv[1]) if argCount >= 2 else 'ts_test2.mp4'
    process_video(inputFile)

# https://www.kaggle.com/code/utkarshsaxenadn/car-detection-yolo-v3-opencv
# https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data
