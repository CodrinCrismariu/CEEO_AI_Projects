# OpenMV AI Face Detection Library

This project demonstrates how to use the AI Face Detection library on an OpenMV camera for face detection and angle tracking. The included example (`main_example.py`) shows how to capture images, detect faces, draw bounding boxes and keypoints, and calculate the angular offset of each detected face relative to the camera’s optical center.


## Overview

The AI Face Detection library (provided as `AI_FaceDetection.py`, `BlazeFaceDetector.py`, `BlazeFaceUtils`) implements google mediapipe face detection features optimized for the OpenMV camera. It automatically detects faces (up to eight per frame) in images, returns each face’s bounding box and facial keypoints, and even calculates the horizontal and vertical angles (relative to the camera center) for applications like head tracking.

## Requirements

- **OpenMV Camera:** OpenMV H7 model
- **OpenMV IDE:** The official development environment for OpenMV cameras. Download it from [openmv.io](https://openmv.io/pages/download).
- **Firmware File:** A `firmware.bin` file used to update your camera’s firmware.
  
## Library Usage

Each detection is a dictionary containing:

"bounding_box": Coordinates and dimensions of the face region.

"left_eye", "right_eye", "nose", "mouth", "left_ear", "right_ear": Coordinates for the facial keypoints.

"confidence": A confidence score indicating detection quality.

### To compute the angular position of a face relative to the camera’s center, use:
angle_x, angle_y = detector.angle_relative_to_camera(detection)
Example Workflow
The main_example.py provided in this project demonstrates a complete workflow:

Initialize and configure the sensor.

Capture an image.

Detect faces.

Draw bounding boxes and keypoints on the detected faces.

Calculate and print confidence scores and angle offsets.

You can run this example directly from the OpenMV IDE after connecting your camera.

Flashing Firmware to Your OpenMV Camera
Updating your camera’s firmware is essential for ensuring optimal performance and compatibility with the latest libraries and features. Follow these steps to flash a new firmware (firmware.bin) using the OpenMV IDE:

Step 1: Download and Install OpenMV IDE
Download: Visit openmv.io to download the latest version of the OpenMV IDE.

Install: Follow the installation instructions for your operating system.

Step 2: Connect Your OpenMV Camera
Connect your OpenMV camera to your computer via USB.

Launch the OpenMV IDE. The IDE should automatically detect your connected camera.

Step 3: Flash the Firmware
In the OpenMV IDE, navigate to the Tools menu and select Flash Firmware (or click the flash icon in the toolbar).

A dialog box will appear prompting you to select the firmware file. Browse to the location of your firmware.bin file and select it.

Confirm the action and allow the IDE to begin flashing the firmware. Do not disconnect your camera during this process.

Once completed, the camera will automatically reboot with the updated firmware.

Step 4: Verify the Firmware Update
After reboot, the OpenMV IDE should reconnect to your camera.

Verify the new firmware version using the IDE's console or check the camera settings.