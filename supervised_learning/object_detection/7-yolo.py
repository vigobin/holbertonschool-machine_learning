#!/usr/bin/env python3
"""Predict Images"""

import cv2
import numpy as np
import os
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
                            tw) / self.model.input.shape[1]
                        bh = self.anchors[i][anchor][1] * np.exp(
                            th) / self.model.input.shape[2]

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

        return boxes, box_confidences, box_class_probs

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
            box_scores_per_class = box_conf * box_class_prob
            box_class = np.argmax(box_scores_per_class, axis=-1)
            box_score = np.max(box_scores_per_class, axis=-1)

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
        sorted_indices = np.argsort(box_scores)[::-1]
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        while len(sorted_indices) > 0:
            # Get the index of the box with the highest score
            max_score_index = sorted_indices[0]

            # Append the corresponding box, class, and score to the final lists
            box_predictions.append(filtered_boxes[max_score_index])
            predicted_box_classes.append(box_classes[max_score_index])
            predicted_box_scores.append(box_scores[max_score_index])

            # Compute IoU for the current box with all other boxes
            iou = self.compute_iou(filtered_boxes[max_score_index],
                                   filtered_boxes[sorted_indices[1:]])

            # Find indices of boxes with IoU less than NMS threshold
            overlapping_indices = np.where(iou <= self.nms_t)[0]

            # Update the sorted_indices list by removing overlapping boxes
            sorted_indices = sorted_indices[overlapping_indices + 1]

        return np.array(box_predictions), np.array(
            predicted_box_classes), np.array(predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        folder_path: a string representing the path to the folder holding
            all the images to load
        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', 'jpeg')):
                images.append(cv2.imread(folder_path + '/' + filename))
                image_paths.append(folder_path + '/' + filename)

        return images, image_paths

    def preprocess_images(self, images):
        """
        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model Note: this
                    can vary by model
                input_w: the input width for the Darknet model Note: this
                    can vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
                original height and width of the images
                2 => (image_height, image_width)
        """
        pimages = []
        image_shapes = []

        for image in images:
            h, w = image.shape[:2]
            resized_image = cv2.resize(image,
                                       (self.model.input.shape[1],
                                        self.model.input.shape[2]),
                                       interpolation=cv2.INTER_CUBIC)
            normalized_image = resized_image / 255.0

            pimages.append(normalized_image)
            image_shapes.append((h, w))

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored
        Displays the image with all boundary boxes, class names, and
            box scores.
        """
        for box, box_class, box_score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[box_class]
            rounded_score = round(box_score, 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f'{class_name} {rounded_score}', (
                x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        folder_path: a string representing the path to the folder holding
            all the images to predict.
        All image windows should be named after the corresponding image
            filename without its full path.
        Displays all images using the show_boxes method.
        Returns: a tuple of (predictions, image_paths):
            predictions: a list of tuples for each image of
                (boxes, box_classes, box_scores)
            image_paths: a list of image paths corresponding to each
                prediction in predictions.
        """
        predictions = []
        image_paths = []

        # Load images and their file names using the load_images function
        images, image_files = self.load_images(folder_path)

        # Iterate through each image and its corresponding file name
        for image, image_file in zip(images, image_files):
            # Preprocess the image for YOLO model
            pimage, _ = self.preprocess_images([image])

            # Predict using the YOLO model
            outputs = self.model.predict(pimage)

            # Process outputs to obtain bounding boxes, confidences
            #   and class probabilities
            boxes, box_confidences, box_class_probs = self.process_outputs(
                outputs, image.shape[:2])

            # Filter boxes based on confidence threshold and get corresponding
            #   class indices and scores
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)

            # Apply non-max suppression to get final predictions
            box_predictions, predicted_box_classes, predicted_box_scores =\
                self.non_max_suppression(filtered_boxes, box_classes,
                                         box_scores)

            # Show the image with bounding boxes, class names, and scores
            self.show_boxes(image.copy(), box_predictions,
                            predicted_box_classes, predicted_box_scores,
                            image_file)

            # Append predictions and image path to respective lists
            predictions.append((box_predictions, predicted_box_classes,
                                predicted_box_scores))
            image_paths.append(os.path.join(folder_path, image_file))

        # Return the list of predictions and corresponding image paths
        return predictions, image_paths
