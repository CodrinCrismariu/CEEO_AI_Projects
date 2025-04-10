import time, ml, math
from BlazeFaceUtils import SsdAnchorsCalculatorOptions, gen_anchors

# Constants
KEY_POINT_SIZE = 6      # Number of facial keypoints per detection.
MAX_FACE_NUM = 8      # Maximum number of faces to keep after NMS.

#------------------------------------------------------------------------------
# BlazeFace Detector class using the ml module
#------------------------------------------------------------------------------
class BlazeFaceDetector:
    def __init__(self, model_path,
                 score_threshold=0.7, iou_threshold=0.3):
        self.score_threshold = score_threshold  # Detection probability threshold.
        self.iou_threshold = iou_threshold      # IoU threshold for non-max suppression.
        self.fps = 0
        self.last_time = time.ticks_ms()
        self.frame_counter = 0

        # Define the model input dimensions.
        self.input_width = 128
        self.input_height = 128

        # Load the TFLite model using the ml module.
        self.model = ml.Model(model_path)

        # Generate anchors for the 896 detections.
        self.anchors = self.generateAnchors()

    #------------------------------------------------------------------------------
    # Generate anchors similar to the original BlazeFace implementation.
    # This example uses:
    #   - input size: 128
    #   - min_scale: 0.1484375, max_scale: 0.75
    #   - 4 layers with strides: [8, 16, 16, 16]
    #   - 2 anchors per grid cell (yielding 896 anchors total)
    #------------------------------------------------------------------------------
    def generateAnchors(self):
        # Replicate the original SSD anchor calculator.
        options = SsdAnchorsCalculatorOptions(
            input_size_width=128,
            input_size_height=128,
            min_scale=0.1484375,
            max_scale=0.75,
            num_layers=4,
            feature_map_width=[],  # Let the function compute feature map sizes.
            feature_map_height=[],
            strides=[8, 16, 16, 16],
            aspect_ratios=[1.0],
            anchor_offset_x=0.5,
            anchor_offset_y=0.5,
            reduce_boxes_in_lowest_layer=False,
            interpolated_scale_aspect_ratio=1.0,
            fixed_anchor_size=False
        )
        return gen_anchors(options)

    #------------------------------------------------------------------------------
    # Prepare the input image.
    # Since the sensor is now configured to capture 128x128 images directly,
    # we only need to convert the image to an ndarray and add a batch dimension.
    #------------------------------------------------------------------------------
    def prepare_input(self, img):
        # Create a Normalization object that maps input pixel values into the range [-1, 1].
        # For simple linear scaling, you can leave mean and stdev at their default values.
        norm = ml.preprocessing.Normalization(scale=(-1, 1))
        arr = norm(img)
        # The resulting array should have shape (128, 128, 3) with values between -1 and 1.
        # Reshape to add the batch dimension to match (1, 128, 128, 3)
        return arr

    #------------------------------------------------------------------------------
    # Decode raw model outputs into a list of detections.
    # Each detection is a tuple:
    #   (x, y, w, h, score, keypoints)
    # where x, y, w, h are normalized (0 to 1) with x,y as the top-left corner,
    # and keypoints is a list of (x,y) tuples.
    #------------------------------------------------------------------------------
    def decode_detections(self, boxes, scores):
        detections = []
        num_detections = scores.shape[0]  # Should be 896.
        for i in range(num_detections):
            # Each score is of shape (1,); apply sigmoid to convert to probability.
            raw_score = scores[i][0]
            score = 1.0 / (1.0 + math.exp(-raw_score))
            if score < self.score_threshold:
                continue

            # Extract the raw bounding box predictions.
            sx = boxes[i][0]
            sy = boxes[i][1]
            w  = boxes[i][2]
            h  = boxes[i][3]
            anchor = self.anchors[i]
            # Decode center coordinates.
            cx = (sx + anchor.x_center * self.input_width) / self.input_width
            cy = (sy + anchor.y_center * self.input_height) / self.input_height
            # Normalize width and height.
            w_norm = w / self.input_width
            h_norm = h / self.input_height

            # Convert from center coordinates to top-left corner.
            x1 = cx - w_norm * 0.5
            y1 = cy - h_norm * 0.5

            # Decode facial keypoints.
            keypoints = []
            for j in range(KEY_POINT_SIZE):
                lx = boxes[i][4 + 2*j]
                ly = boxes[i][4 + 2*j + 1]
                kp_x = (lx + anchor.x_center * self.input_width) / self.input_width
                kp_y = (ly + anchor.y_center * self.input_height) / self.input_height
                keypoints.append((kp_x, kp_y))

            detections.append((x1, y1, w_norm, h_norm, score, keypoints))
        return detections

    #------------------------------------------------------------------------------
    # Compute Intersection over Union (IoU) between two detections.
    # Each detection is (x, y, w, h, score, keypoints) with (x,y) as top-left.
    #------------------------------------------------------------------------------
    def iou(self, det1, det2):
        x1, y1, w1, h1 = det1[0], det1[1], det1[2], det1[3]
        x2, y2, w2, h2 = det2[0], det2[1], det2[2], det2[3]
        # Determine the (x, y)-coordinates of the intersection rectangle.
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area

    #------------------------------------------------------------------------------
    # Apply non-max suppression to reduce overlapping detections.
    #------------------------------------------------------------------------------
    def non_max_suppression(self, detections):
        # Sort detections by score (highest first).
        detections = sorted(detections, key=lambda d: d[4], reverse=True)
        final_detections = []
        while detections:
            best = detections.pop(0)
            final_detections.append(best)
            detections = [d for d in detections if self.iou(best, d) < self.iou_threshold]
            if len(final_detections) >= MAX_FACE_NUM:
                break
        return final_detections

    #------------------------------------------------------------------------------
    # Update the FPS counter.
    #------------------------------------------------------------------------------
    def update_fps(self):
        self.frame_counter += 1
        current_time = time.ticks_ms()
        dt = time.ticks_diff(current_time, self.last_time)
        if dt > 0:
            self.fps = 1000 // dt
        self.last_time = current_time
        self.frame_counter = 0

    #------------------------------------------------------------------------------
    # Run the full detection pipeline:
    #   1. Preprocess the image.
    #   2. Run inference via ml.Model.predict().
    #   3. Decode raw outputs into detections.
    #   4. Apply non-max suppression.
    #   5. Update FPS.
    # Returns a list of final detections.
    #------------------------------------------------------------------------------
    def detect_faces(self, img):
        # Since the sensor is capturing 128x128 directly, the original image dimensions are 128x128.
        orig_w = img.width()
        orig_h = img.height()

        # Prepare the image for inference.
        inp = self.prepare_input(img)
        # Run inference. predict() requires a list of inputs.
        outputs = self.model.predict([inp])
        # According to our model, outputs[0] is the scores tensor (shape: (1,896,1))
        # and outputs[1] is the boxes tensor (shape: (1,896,16)).
        scores = outputs[0][0]  # Remove the batch dimension → shape (896, 1)
        boxes = outputs[1][0]   # Remove the batch dimension → shape (896, 16)

        # Decode raw outputs into detection candidates.
        detections = self.decode_detections(boxes, scores)
        # Apply non-max suppression to remove overlapping detections.
        final_detections = self.non_max_suppression(detections)
        self.update_fps()
        return final_detections, orig_w, orig_h

    #------------------------------------------------------------------------------
    # Draw detections on the image.
    # Coordinates are normalized (0 to 1), so scale them to the original image size.
    #------------------------------------------------------------------------------
    def draw_detections(self, img, detections, orig_w, orig_h):
        for det in detections:
            x, y, w, h, score, keypoints = det
            # Convert normalized coordinates to pixel values.
            x1 = int(x * orig_w)
            y1 = int(y * orig_h)
            w_px = int(w * orig_w)
            h_px = int(h * orig_h)
            img.draw_rectangle((x1, y1, w_px, h_px), color=(22, 22, 250))
            # img.draw_string(x1, y1 - 6, "%.2f" % score, color=(22, 22, 250))
            # Draw keypoints.
            for kp in keypoints:
                kp_x = int(kp[0] * orig_w)
                kp_y = int(kp[1] * orig_h)
                img.draw_circle(kp_x, kp_y, 2, color=(214, 202, 18))
        return img
