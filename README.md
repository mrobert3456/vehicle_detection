# Vehicle detection

![project_video_clip](./data/output.gif)

# ⚙ How it works

## Vehicle detection overview

1. **Load model** labels
2. **Load network** model
3. **Set the target device** to GPU
4. **Get detection layers** from YOLOv3 network
5. **Get the current frame** from the video.
6. **Get the Region Of Interest** from the retrieved frame
7. **Preprocess frame**
8. **Forward the frame** to the network
9. **Iterate through the detected objects** of each output layer
10. **Get the detected objects bounding indicies, class and confidence score**
11. **Non-maximum suppression** of given bounding boxes
12. **Draw bounding boxes** on the input frame

---
# 📦 Installation

## This Repository

Download this repository by running:

```
git clone https://github.com/mrobert3456/vehicle_detection.git
cd vehicle_detection
```

## YOLOv3 coco dataset
The necessary files (*coco.names*, *yolov3.cfg*, *yolov3.weights*) can be downloaded at:
https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data

## ⚡ Software Dependencies

This project utilizes the following packages:

* Python 3
* OpenCV 2
* Numpy

Install dependencies by running:
```
pip install -r requirements.txt
```

# 🚀 Usage


To produce the output, simply run:

```
python main.py input_file.mp4
```
