import sys
import cv2 as cv
import numpy as np
from utils.image_roi import get_image_roi
from tqdm import tqdm


class YoloVehicleDetector:
    def __init__(self, cfg_path, weights_path, labels_path):
        f = open(labels_path, 'rb')

        self.labels = list(n.decode('UTF-8').replace('\n', ' ').strip() for n in f.readlines())
        self.network = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.probability_minimum = 0.6
        self.overlap_threshold = 0.2

        yolo_layers = self.network.getLayerNames()
        self.output_layers = [yolo_layers[i - 1] for i in self.network.getUnconnectedOutLayers()]
        self.network.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.network.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    def detect_vehicles(self, image):
        """
        Detects vehicles on the image
        :param image: Original image
        :return: Original image with bounding boxes.
        """
        h, w = image.shape[:2]
        blob = self._process_image(image)

        self.network.setInput(blob)
        output_from_network = self.network.forward(self.output_layers)

        # array for detected bounding boxes, confidences and class number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # output layers after feed forward pass
        for result in output_from_network:
            # all detections from current output layer
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if self.labels[class_current] in ["car", "truck",
                                                  "motorbike"] and confidence_current > self.probability_minimum:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Getting top left corner coordinates of bounding box
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        return self._draw_bounding_boxes(image, bounding_boxes, confidences, class_numbers)

    def _process_image(self, image):
        """
        Gets the region of interest from the image, then creates a blob from it
        :param image: image to be processed
        :return: blob that can be passed to the yolo network
        """
        image_roi = get_image_roi(image)
        blob = cv.dnn.blobFromImage(image_roi, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        return blob

    def _draw_bounding_boxes(self, image, bounding_boxes, confidences, class_numbers):
        """
        Draws the bounding boxes around the detected objects
        :param image: image to draw on
        :param bounding_boxes: bounding box coordinates array
        :param confidences: confidence scores array
        :param class_numbers: class index array
        :return: Original image with bounding boxes.
        """

        results = cv.dnn.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.overlap_threshold)

        if len(results) > 0:
            for i in results.flatten():
                # Bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                cv.rectangle(image, (x_min, y_min),
                             (x_min + box_width, y_min + box_height),
                             (0, 255, 0), 2)

                text_box_current = '{}: {:.4f}'.format(self.labels[class_numbers[i]],
                                                       confidences[i])

                cv.putText(image, text_box_current, (x_min, y_min - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image


class ObjectDetector:
    def __init__(self, detector, result_video_path):
        self.detector = detector
        self.frame_array = []

        self.fps_count = 25
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.writer = cv.VideoWriter(result_video_path, fourcc, self.fps_count,
                                     (1280, 720), True)

    def process_video(self, video_path):
        """
        Loads a video then detects vehicles frame by frame.
        :param video_path: Input video path
        :return: Creates a new mp4 video with the detected objects.
        """
        capture = cv.VideoCapture(video_path)
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv.resize(frame, (1280, 720))

            result_img = self.detector.detect_vehicles(frame)
            self.frame_array.append(result_img)

            cv.imshow('frame', result_img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv.destroyAllWindows()

        self._write_video()

    def _write_video(self):
        """
        Creates a new video from self.frame_array.
        :return:
        """
        start = 0
        for i in range(0, len(self.frame_array)):
            if (start + self.fps_count) < len(self.frame_array) - 1:
                end = start + self.fps_count
                for j in tqdm(range(start, end)):
                    self.writer.write(self.frame_array[j])

            start = start + self.fps_count
        self.writer.release()


if __name__ == '__main__':
    argCount = len(sys.argv)
    inputFile = str(sys.argv[1]) if argCount >= 2 else 'ts_test2.mp4'

    path_to_weights = 'data/yolov3.weights'
    path_to_cfg = 'data/yolov3.cfg'
    path_to_labels = 'data/coco.names'

    yolo_detector = YoloVehicleDetector(path_to_cfg, path_to_weights, path_to_labels)
    object_detector = ObjectDetector(yolo_detector, "output.mp4")

    object_detector.process_video(inputFile)

# https://www.kaggle.com/code/utkarshsaxenadn/car-detection-yolo-v3-opencv
# https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data
