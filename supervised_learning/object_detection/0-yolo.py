#!/usr/bin/env python3
"""Initialize Yolo"""

import numpy as np
import tensorflow.keras as K


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
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_classes(self, path):
        """Function to load class names for model"""
        with open(path, 'r') as file:
            classes = file.read().splitlines()
        return classes
