#!/usr/bin/env python3
""" Filter Boxes """

import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Model_path is the path to where a Darknet Keras model is stored.
            Classes_path is the path to where the list of class names used
                for the Darknet model, listed in order of index, can be found.
            Class_t is a float representing the box score threshold for the
                initial filtering step.
            nms_t is a float representing the IOU threshold for
                non-max suppression.
            Anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes:
                outputs is the number of outputs (predictions)
                    made by the Darknet model.
                anchor_boxes is the number of anchor boxes used for
                    each prediction.
                2 => [anchor_box_width, anchor_box_height]
        """
        self.model = load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_classes(self, path):
        """Function to load class names for model"""
        with open(path, 'r') as file:
            classes = file.read().splitlines()
        return classes

    def process_outputs(self, outputs, image_size):
        """Outputs is a list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                    anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width => the height and
                    width of the grid used for the output
                    anchor_boxes => the number of anchor boxes used
                    4 => (t_x, t_y, t_w, t_h)
                    1 => box_confidence
                    classes => class probabilities for all classes
            image_size is a numpy.ndarray containing the image’s original size
                [image_height, image_width]
            Returns a tuple of (boxes, box_confidences, box_class_probs):
                boxes: a list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 4) containing the processed
                    boundary boxes for each output, respectively:
                    4 => (x1, y1, x2, y2)
                    (x1, y1, x2, y2) should represent the boundary box relative
                    to original image
                box_confidences: a list of numpy.ndarrays of shape (
                        grid_height, grid_width, anchor_boxes, 1) containing
                        the box confidences for each output, respectively
                box_class_probs: a list of numpy.ndarrays of shape (
                        grid_height, grid_width, anchor_boxes, classes)
                        containing the box’s class probabilities for each
                        output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., :4])
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])

        for i, box in enumerate(boxes):
            grid_h, grid_w, anchor_boxes, _ = box.shape

            cy = np.indices((grid_h, grid_w, anchor_boxes))[0]
            cx = np.indices((grid_h, grid_w, anchor_boxes))[1]

            tx = (box[..., 0] + cx) / grid_w
            ty = (box[..., 1] + cy) / grid_h
            tw = np.exp(box[..., 2]) * self.anchors[i][
                :, 0] / self.model.input.shape[1].value
            th = np.exp(box[..., 3]) * self.anchors[i][
                :, 1] / self.model.input.shape[2].value

            box[..., 0] = (tx - tw / 2) * image_size[1]
            box[..., 1] = (ty - th / 2) * image_size[0]
            box[..., 2] = (tx + tw / 2) * image_size[1]
            box[..., 3] = (ty + th / 2) * image_size[0]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape (
            grid_height, grid_width, anchor_boxes, 4)containing
            the processed boundary boxes for each output, respectively
        box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing
                the processed box confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing
                the processed box class probabilities for each output.
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4)
                containing all of the filtered bounding boxes
            box_classes: a numpy.ndarray of shape (?,)
                containing the class number that each box in filtered_boxes
                predicts, respectively
            box_scores: a numpy.ndarray of shape (?)
                containing the box scores for each box in filtered_boxes
        """
        boxes_filtered = []
        scores = []
        classes = []

        for box, confidences, class_probs in zip(boxes, box_confidences,
                                                 box_class_probs):
            box_scores = confidences * class_probs
            box_classes = np.argmax(box_scores, axis=-1)
            box_class_scores = np.max(box_scores, axis=-1)

            filtering_mask = box_class_scores >= self.class_t

            filtered_boxes = box[filtering_mask]
            filtered_scores = box_class_scores[filtering_mask]
            filtered_classes = box_classes[filtering_mask]

            boxes_filtered.append(filtered_boxes)
            scores.append(filtered_scores)
            classes.append(filtered_classes)

        return boxes_filtered, scores, classes
