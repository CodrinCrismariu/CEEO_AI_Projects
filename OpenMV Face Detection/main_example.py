# Main Face detection example - By: Codrin - Thu Apr 10 2025

import sensor
from AI_FaceDetection import AI_FaceDetection

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.B128X128)
sensor.skip_frames(time=2000)

# example of use of face detection
detector = AI_FaceDetection()

while True:
    img = sensor.snapshot()
    # Preffered input image is 128x128 RGB565 image
    # but the function will resize if necessary.
    detections = detector.detect_faces(img)

    # detections represents an array of all of the detected faces
    # up to 8 detections

    for detection in detections:
        # Draw the bounding box around the face
        img.draw_rectangle(detection["bounding_box"], color=(255, 0, 0))

        # Draw the keypoints
        img.draw_circle(detection["left_eye"][0], detection["left_eye"][1], 2, color=(255, 0, 0))
        img.draw_circle(detection["right_eye"][0], detection["right_eye"][1], 2, color=(255, 0, 0))
        img.draw_circle(detection["nose"][0], detection["nose"][1], 2, color=(255, 0, 0))
        img.draw_circle(detection["mouth"][0], detection["mouth"][1], 2, color=(255, 0, 0))
        img.draw_circle(detection["left_ear"][0], detection["left_ear"][1], 2, color=(255, 0, 0))
        img.draw_circle(detection["right_ear"][0], detection["right_ear"][1], 2, color=(255, 0, 0))

        # Print the confidence score
        print("Confidence: {:.2f}".format(detection["confidence"]))


        # Get Angle relative to camera
        # Return angle position of the face relative to the center of the image
        # Can be used for head tracking
        angle_x, angle_y = detector.angle_relative_to_camera(detection)

        print("Angle X: {:.2f}, Angle Y: {:.2f}".format(angle_x, angle_y))



