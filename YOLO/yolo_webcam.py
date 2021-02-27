import cv2
from helpers import *
from image import *
from labels import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow import expand_dims


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


# define anchors, class_threshold
anchors = [[116, 90, 156, 198, 373, 326], [
    30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
class_threshold = 0.6
nms_threshold = 0.5
input_w, input_h = 416, 416

# load model
model = load_model('model.h5')

video_capture = cv2.VideoCapture(0)
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (input_w, input_h))
    small_frame = small_frame.astype('float32')
    small_frame /= 255.0
    # add a dimension so that we have one sample
    small_frame = expand_dims(small_frame, 0)
    boxes = []
    if process_this_frame:
        yhat = model.predict(small_frame)
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], anchors[i],
                                   class_threshold, input_h, input_w)

        # scale bounding boxes
        correct_yolo_boxes(
            boxes, frame.shape[0], frame.shape[1], input_h, input_w)

        # suppress non-maximal boxes (overlaped bounding boxes)
        do_nms(boxes, nms_threshold)

        v_boxes, v_labels, v_scores = get_boxes(
            boxes, labels, thresh=class_threshold)
    process_this_frame = not process_this_frame
    for i, box in enumerate(v_boxes):
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # draw label
        # cv2.rectangle(frame, (x1, bottom - 35),
        #               (right, bottom), (0, 0, 255), cv2.FILLED)
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (x1, y1),
                    font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Hit 'q' on the keyboard to quit!


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
