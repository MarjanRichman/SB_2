import cv2
import numpy as np
import glob
import random
from mean_average_precision import MetricBuilder


def convert2mask(mt, shape):
    # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

    t = np.zeros([shape, shape])
    # print(t.shape)
    for m in mt:
        # print("M ", m)
        x, y, w, h = m
        # print(x, y, w, h)
        cv2.rectangle(t, (x, y), (x+w, y+h), 1, -1)
    return t


def prepare_for_detection(prediction, ground_truth):
        # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

        if len(prediction) == 0:
            return [], []
        # Large enough size for base mask matrices:
        shape = 2*max(np.max(prediction), np.max(ground_truth))
        p = convert2mask(prediction, shape)
        gt = convert2mask(ground_truth, shape)

        return p, gt


def iou_compute(p, gt):
        # Computes Intersection Over Union (IOU)
        if len(p) == 0:
            return 0
        intersection = np.logical_and(p, gt)
        union = np.logical_or(p, gt)

        iou = np.sum(intersection) / np.sum(union)
        return iou


gt_path = glob.glob(r"C:/Users/Marjan/Desktop/SB 2. assignment/data/ears/annotations/detection/"
                        r"test_YOLO_format/*.txt")
idx = 0

mAP_gt = []


def get_gt():
    boxes = []
    with open(gt_path[idx]) as f:
        lines = f.readlines()
        for line in lines:
            gx = line.split(' ')
            gx.pop(5)
            gx.pop(0)
            for i in range(0, len(gx)):
                gx[i] = int(gx[i])
            boxes.append(gx)
            xmin, ymin, wid, hei = gx
            mAP_gt.append([xmin, ymin, xmin + wid, ymin + hei, 0, 0, 0])
    return boxes


if __name__ == '__main__':

    # Load Yolo
    # net = cv2.dnn.readNet("yolov3_training_last_(prvi model, 1542 epoh. neokrnjen dataset.weights", "yolov3_testing.cfg")
    # net = cv2.dnn.readNet("yolo_training_last.weights", "yolo_testing.cfg")
    # net = cv2.dnn.readNet("yolov2-training_last.weights", "yolov2_testing.cfg")
    # net = cv2.dnn.readNet("yolov3-training_last.weights", "yolov3_testing.cfg")
    net = cv2.dnn.readNet("yolov3-tiny-training_last.weights", "yolov3-tiny-testing.cfg")
    # net = cv2.dnn.readNet("yolov4-tiny-training_last.weights", "yolov4-tiny_testing.cfg")

    # Name custom object
    classes = ["Ear"]

    # Images path
    images_path = glob.glob(r"C:/Users/Marjan/Desktop/SB 2. assignment/data/ears/test/*.png")
    # images_path = glob.glob(r"C:/Users/Marjan/Desktop/SB 2. assignment/preprocessing/*.png")


    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    mAP_pred = []

    # Insert here the path of your images
    # loop through all the images

    loop = 0

    iou_total = []

    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape

        # Loading ground truth
        gt = get_gt()
        gt_idx = 0
        idx += 1

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    # print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)
        # print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        pred = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                pred.append(boxes[i])
                # [xmin, ymin, xmax, ymax, class_id, confidence]
                mAP_pred.append([x, y, x + w, y + h, 0, confidences[i]])
        p, gt = prepare_for_detection(pred, gt)
        iou = iou_compute(p, gt)
        print("CURRENT ", loop, iou)
        iou_total.append(iou)
        loop += 1
        cv2.imshow("Image", img)
        key = cv2.waitKey(0)
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    metric_fn.add(np.array(mAP_pred), np.array(mAP_gt))

    # compute IOU metric
    print("IOU: ", sum(iou_total) / len(iou_total))

    # compute PASCAL VOC metric
    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

    # compute metric COCO metric
    print(
        f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
    cv2.destroyAllWindows()