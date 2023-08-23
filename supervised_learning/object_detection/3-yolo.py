#!/usr/bin/env python3
"""Non-max Suppression"""

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
            Returns a tuple of (
                processed_boxes,box_confidences, box_class_probs):
             processed_boxes: a list of numpy.ndarrays of shape (grid_height,
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
        # Initialize lists for processed data from outputs
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Extract data for boundary boxes using slicing from each output,
        # transform confidence scores and class probabilities,
        # through a sigmoid activation function.
        for output in range(len(outputs)):
            boxes.append(outputs[output][..., :4])
            box_confidences.append(self.sigmoid(outputs[output][..., 4:5]))
            box_class_probs.append(self.sigmoid(outputs[output][..., 5:]))

        # Iterate through list of boxes to keep track of the corresponding
        #  anchor configuration.
        processed_boxes = []
        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape
            processed_box = np.zeros_like(box)

            # Iterate through each cell in the grid and each anchor box
            #   configuration. For each cell and anchor, it calculates the
            #   transformed bounding box coordinates.
            for row in range(grid_height):
                for col in range(grid_width):
                    for anchor in range(anchor_boxes):
                        tx, ty, tw, th = box[row, col, anchor, :4]
                        # Predicted center of the box (cx and cy) is
                        #   adjusted by applying the sigmoid function.
                        cx = (col + self.sigmoid(tx)) / grid_width
                        cy = (row + self.sigmoid(ty)) / grid_height
                        # Predicted width and height of the box (bw and bh)
                        #   are adjusted using the anchor box dimensions and
                        #   the exponential transformation of tw and th.
                        bw = self.anchors[i][anchor][0] * np.exp(
                            tw) / self.model.input.shape[1].value
                        bh = self.anchors[i][anchor][1] * np.exp(
                            th) / self.model.input.shape[2].value

                        #  Transformed bounding box coordinates are scaled
                        #   to the original image size.
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
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

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
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_class_prob in (zip(
                boxes, box_confidences, box_class_probs)):
            # Confidence * probability gives the box score.
            box_scores_per_class = box_conf * box_class_prob
            # find class predictions that exceed threshold.
            box_class = np.argmax(box_scores_per_class, axis=-1)
            box_score = np.max(box_scores_per_class, axis=-1)

            # Filter box score below threshold.
            mask = box_score >= self.class_t
            filtered_boxes.extend(box[mask])
            box_classes.extend(box_class[mask])
            box_scores.extend(box_score[mask])

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

    def compute_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        iou = intersection_area / (box1_area + box2_area - intersection_area)
        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
            filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
            for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
            each box in filtered_boxes, respectively
        Returns a tuple of
            (box_predictions, predicted_box_classes, predicted_box_scores):
        box_predictions: a numpy.ndarray of shape (?, 4) containing all
            of the predicted bounding boxes ordered by class and box score
        predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions ordered by class and box score.
        predicted_box_scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions ordered by class and box score
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in set(box_classes):
            class_mask = np.where(box_classes == cls)
            class_boxes = filtered_boxes[class_mask]
            class_box_scores = box_scores[class_mask]

            while len(class_boxes) > 0:
                # Get the index of the box with the highest score
                max_score_index = np.argmax(class_box_scores)

                # Append the corresponding box, class, and score to the
                #   final lists
                box_predictions.append(class_boxes[max_score_index])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_box_scores[max_score_index])

                # Remove bounding box and confidence score
                #   for the highest index.
                class_boxes = np.delete(class_boxes, max_score_index, axis=0)
                class_box_scores = np.delete(class_box_scores,
                                             max_score_index, axis=0)

                # Check that there are no more bounding boxes to remove
                #  so that all duplicates are removed.
                if len(class_boxes) == 0:
                    break

                # Compute IoU for the current box with all other boxes
                iou = self.compute_iou(box_predictions[-1], class_boxes)

                # Find indices of boxes with IoU less than NMS threshold
                overlapping_indices = iou < self.nms_t

                # Update the sorted_indices list by removing overlapping boxes
                class_boxes = class_boxes[overlapping_indices]
                class_box_scores = class_box_scores[overlapping_indices]

        return np.array(box_predictions), np.array(
            predicted_box_classes), np.array(predicted_box_scores)
