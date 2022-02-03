import os
import cv2
import time
import zipfile
import progressbar
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# Prevent tensorflow from using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras

# Define Paths:
# yolo model folder:
res_path = "resources/"
weights_Path = res_path + "yolov3.weights"
config_Path = res_path + "yolov3.cfg"
labels_Path = res_path + "coco.names"

accepted_confidence = 0.3
threshold = 0.3

name_of_model = "estimator_model"
zip_model_path = res_path + name_of_model + ".zip"
h5_model_path = res_path + name_of_model + ".h5"

pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()
    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

# Download to colab:
def temporary_downloader(path, link):
    print("\nDownloading", path)
    urllib.request.urlretrieve(link, path, show_progress)

# Create yolo model folder and download files:
if not os.path.exists(res_path):
    os.mkdir(res_path)
# download yolov3.cfg:
if not os.path.isfile(config_Path):
    temporary_downloader(config_Path, "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
# download yolov3.weights:
if not os.path.isfile(weights_Path):
    temporary_downloader(weights_Path, "https://pjreddie.com/media/files/yolov3.weights")
# download coco.names:
if not os.path.isfile(labels_Path):
    temporary_downloader(labels_Path, "https://raw.githubusercontent.com/pjreddie/darknet/a028bfa0da8eb96583c05d5bd00f4bfb9543f2da/data/coco.names")

# Download estimation model:
if not os.path.isfile(h5_model_path):
    temporary_downloader(zip_model_path, "https://s20.picofile.com/d/8447340318/e94262e5-ca04-42f9-84cc-a588c7404960/estimator_model.zip")
    with zipfile.ZipFile(zip_model_path) as zf:
        zf.extractall(res_path)

# load the COCO class labels our YOLO model was trained on
LABELS = open(labels_Path).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_Path, weights_Path)

# load our input image and grab its spatial dimensions
# Image to test:
image_path = "test.png"
rgb = cv2.imread(image_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
rgb_display = np.copy(rgb)
(H, W) = rgb.shape[:2]
# determine only the *output* layer names that we need from YOLO
# ln stands for LayerName
ln = net.getLayerNames()
if net.getUnconnectedOutLayers().ndim == 1:
  ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
else:
  ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(rgb, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

t = time.time()
layerOutputs = net.forward(ln)
print("[INFO] YOLO processes time: {:.6f} sec".format(time.time() - t))

# initialize
boxes = []
confidences = []
classIDs = []
# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
		# filter out weak predictions by ensuring the detected probability is greater than the minimum probability
        # extract the class ID and confidence (i.e., probability) of the current object detection
        if confidence > accepted_confidence:
            # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
            # # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

# ESTIMATOR output:
# load trained model:
estimator = keras.models.load_model(h5_model_path)

# Normal the input images:
normalizer = lambda input_image: (input_image / 255)
# Reshape the input images using tensorflow:
tf_resizer = lambda input_image: tf.image.resize(input_image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Reshape the input images using openCV:
resizer = lambda input_image: cv2.resize(input_image, (128, 128))

# prepare RGB image for estimation:
rgb_in = normalizer(rgb)
rgb_in = resizer(rgb_in)
# Estimation of depth
es_out = estimator(rgb_in[tf.newaxis, ...], training=False)
es_out = cv2.resize(np.array(es_out).reshape(128, 128), (rgb.shape[1], rgb.shape[0]))

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        es_box_mean = np.mean(es_out[y:y+h, x:x+w])
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 2)
        text = "{}: Conf={:.4f}, Depth={:.2f}".format(LABELS[classIDs[i]], confidences[i], es_box_mean)
        cv2.putText(rgb, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Plot a sample image:
fig, axarr = plt.subplots(2, 3, figsize=(16,12))
# Plot RGB image:
axarr[0,0].set_title('Original image')
axarr[0,0].axis('off')
axarr[0,0].imshow(rgb_display)
# Plot depth estimated image:
axarr[1,0].set_title('Depth estimated image')
axarr[1,0].axis('off')
axarr[1,0].imshow(es_out, cmap=plt.get_cmap('gray'))
# Plot result image:
axarr[0,1] = plt.subplot2grid(shape=(2, 3), loc=(0, 1), rowspan=2, colspan=2)
axarr[0,1].set_title('Object Detection + Depth Estimation result')
axarr[0,1].axis('off')
axarr[0,1].imshow(rgb)
plt.show()