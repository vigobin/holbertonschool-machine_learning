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
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., :4])
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])

        processed_boxes = []
        for i, box in enumerate(boxes):
            grid_h, grid_w, anchor_boxes, _ = box.shape
            processed_box = np.zeros_like(box)

            for row in range(grid_h):
                for col in range(grid_w):
                    for anchor in range(anchor_boxes):
                        tx, ty, tw, th = box[row, col, anchor, :4]
                        cx = (col + self.sigmoid(tx)) / grid_w
                        cy = (row + self.sigmoid(ty)) / grid_h
                        bw = self.anchors[i][anchor][0] * np.exp(
                            tw) / self.model.input.shape[1]
                        bh = self.anchors[i][anchor][1] * np.exp(
                            th) / self.model.input.shape[2]

                        x1 = (cx - bw / 2) * image_size[1]
                        y1 = (cy - bh / 2) * image_size[0]
                        x2 = (cx + bw / 2) * image_size[1]
                        y2 = (cy + bh / 2) * image_size[0]

                        processed_box[row, col, anchor, 0] = x1
                        processed_box[row, col, anchor, 1] = y1
                        processed_box[row, col, anchor, 2] = x2
                        processed_box[row, col, anchor, 3] = y2

            processed_boxes.append(processed_box)

        return processed_boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
