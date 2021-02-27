'''
YOLOv3 - object detection
Building object detection base on pretrained model (DarkNet code base on the MSCOCO dataset)
- model architechture DarkNet (originally loosely based on the VGG-16 model.)
'''

import numpy
from helpers import *
from image import *
from labels import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_model():
    # define model
    model = make_yolov3_model()

    # load the model weights
    weight_reader = WeightReader('yolov3.weights')

    # set the model weights into the model
    weight_reader.load_weights(model)

    # save the model to file
    model.save('model.h5')


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes = []
    v_labels = []
    v_scores = []

    for box in boxes:
        for i in range(len(labels)):
            # qualify by thresh
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# load model
model = load_model('model.h5')

# define the expected input shape for the model
input_w, input_h = 416, 416

# load image
image, image_w, image_h = load_image_pixels('zebra.jpg', (input_w, input_h))

# make prediction
yhat = model.predict(image)

# summarize the shape of the list of arrays
print([a.shape for a in yhat])

# define anchors, class_threshold
anchors = [[116, 90, 156, 198, 373, 326], [
    30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
class_threshold = 0.6
nms_threshold = 0.5
boxes = []
for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i],
                           class_threshold, input_h, input_w)

# scale bounding boxes
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

# suppress non-maximal boxes (overlaped bounding boxes)
do_nms(boxes, nms_threshold)

# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)


# Draw all bounding boxes
def draw_boxes(file_name, v_boxes, v_labels, v_scores):
    for i, box in enumerate(v_boxes):
        img = plt.imread(file_name)
        plt.imshow(img)
        ax = plt.gca()

        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        plt.text(x1, y1, label, color='blue', weight='bold')
    plt.show()


draw_boxes('zebra.jpg', v_boxes, v_labels, v_scores)
