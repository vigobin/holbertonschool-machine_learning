#!/usr/bin/env python3
"""Process Outputs"""

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
        # Initialize lists for processed data
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Extract data for boundary boxes using slicing from each output
        for output in outputs:
            boxes.append(output[..., :4])
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])
        for i, (box_conf, box_class_prob) in enumerate(zip(box_confidences,
                                                           box_class_probs)):
            grid_h, grid_w, anchor_boxes, _ = box_conf.shape

            cy = np.indices((grid_h, grid_w, anchor_boxes))[0]
            cx = np.indices((grid_h, grid_w, anchor_boxes))[1]

            tx = (boxes[i][..., 0] + cx) / grid_w
            ty = (boxes[i][..., 1] + cy) / grid_h
            tw = np.exp(boxes[i][..., 2]) * self.anchors[i][
                :, 0] / self.model.input.shape[1]
            th = np.exp(boxes[i][..., 3]) * self.anchors[i][
                :, 1] / self.model.input.shape[2]

            boxes[i][..., 0] = (tx - tw / 2) * image_size[1]
            boxes[i][..., 1] = (ty - th / 2) * image_size[0]
            boxes[i][..., 2] = (tx + tw / 2) * image_size[1]
            boxes[i][..., 3] = (ty + th / 2) * image_size[0]

            box_confidences[i] = self.sigmoid(box_conf)
            box_class_probs[i] = self.sigmoid(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))
