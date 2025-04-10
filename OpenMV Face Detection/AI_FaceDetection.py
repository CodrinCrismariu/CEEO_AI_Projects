from BlazeFaceDetector import BlazeFaceDetector
import gc
import math

class AI_FaceDetection:
    def __init__(self):
        self.detector = BlazeFaceDetector(model_path="face_detection_front",
                                          score_threshold=0.7,
                                          iou_threshold=0.3)

    # Function expects RGB 128x128 image but will resize if necessary
    def detect_faces(self, img):
        self.orig_width = img.width()
        self.orig_height = img.height()

        if img.width() != 128 or img.height() != 128:
            # Compute scaling factors so that the resized image is 128x128.
            scale_x = 128 / img.width()
            scale_y = 128 / img.height()

            # Create a new RGB565 image scaled to 128x128 and load it directly into the frame buffer.
            img = img.to_rgb565(x_scale=scale_x, y_scale=scale_y, copy_to_fb=True)

        # Run the detection pipeline on the resized image.
        detections = self.detector.detect_faces(img)
        gc.collect()

        final_detections = []

        for det in detections[0]:
            x, y, w, h, score, keypoints = det

            x = int(x * self.orig_width)
            y = int(y * self.orig_height)

            w = int(w * self.orig_width)
            h = int(h * self.orig_height)

            left_eye_x, left_eye_y = keypoints[0]
            right_eye_x, right_eye_y = keypoints[1]
            nose_x, nose_y = keypoints[2]
            mouth_x, mouth_y = keypoints[3]
            left_ear_x, left_ear_y = keypoints[4]
            right_ear_x, right_ear_y = keypoints[5]

            left_eye_x = int(left_eye_x * self.orig_width)
            left_eye_y = int(left_eye_y * self.orig_height)

            right_eye_x = int(right_eye_x * self.orig_width)
            right_eye_y = int(right_eye_y * self.orig_height)

            nose_x = int(nose_x * self.orig_width)
            nose_y = int(nose_y * self.orig_height)

            mouth_x = int(mouth_x * self.orig_width)
            mouth_y = int(mouth_y * self.orig_height)

            left_ear_x = int(left_ear_x * self.orig_width)
            left_ear_y = int(left_ear_y * self.orig_height)

            right_ear_x = int(right_ear_x * self.orig_width)
            right_ear_y = int(right_ear_y * self.orig_height)

            final_detections.append({
                "bounding_box": (x, y, w, h),
                "confidence": score,
                "keypoints": keypoints,
                "left_eye": (left_eye_x, left_eye_y),
                "right_eye": (right_eye_x, right_eye_y),
                "nose": (nose_x, nose_y),
                "mouth": (mouth_x, mouth_y),
                "left_ear": (left_ear_x, left_ear_y),
                "right_ear": (right_ear_x, right_ear_y),
            })

        return final_detections

    def angle_relative_to_camera(self, detection, hfov=70.8, vfov=55.6):
        cx = self.orig_width / 2.0
        cy = self.orig_height / 2.0

        # Compute focal lengths in pixel units based on the field-of-view.
        # Conversion: tan(HFOV/2 in radians) = (cx / f_x)
        fx = cx / math.tan(math.radians(hfov / 2))
        fy = cy / math.tan(math.radians(vfov / 2))

        # Pixel offsets from the image center
        dx = detection["bounding_box"][0] + detection["bounding_box"][2] / 2 - cx
        dy = detection["bounding_box"][1] + detection["bounding_box"][3] / 2 - cy

        # Calculate the angular offsets in radians then convert to degrees.
        angle_x = math.degrees(math.atan(dx / fx))
        angle_y = math.degrees(math.atan(dy / fy))

        return angle_x, angle_y
