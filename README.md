# Haar Cascade Object Detection using OpenCV

This repository contains a Jupyter notebook demonstrating various applications of Haar Cascade classifiers for object detection, primarily focusing on face, eye, and full-body detection in images and videos.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Features](#features)
4. [Usage](#usage)
5. [Functions](#functions)
6. [Detailed Explanation of the Functions](#explanation)

## Introduction

This project showcases the implementation of Haar Cascade classifiers for detecting human features in static images and video streams. It utilizes OpenCV's pre-trained Haar Cascade models to identify faces, eyes, and full bodies.

## Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib
- Requests

## Features

- Face detection in images
- Face and eye detection in images
- Full-body detection in images
- Face detection in video streams
- Full-body detection in video streams
- Utility functions for drawing bounding boxes and text labels

## Usage

The notebook is divided into several sections, each demonstrating a different aspect of object detection:

1. Face detection from images
2. Face and eye detection from images
3. Human full-body detection from images
4. Face detection from videos
5. Human full-body detection from videos

Each section contains functions that can be called with appropriate parameters to perform the desired detection task.

## Functions

### Key functions include:

1. `detect_face(image_path, save_fig, filename)`: Detects faces in a given image.
2. `detect_face_and_eye(image_path, save_fig, filename)`: Detects faces and eyes in a given image.
3. `detect_human_body(image_path, scale_factor, min_neighbours, save_fig, filename)`: Detects full bodies in a given image.
4. `detect_faces_from_video(video_path, scale_factor, min_neighbors, resize_factor, save_video, filename)`: Detects faces in a video stream.
5. `detect_human_body_from_video(video_path, scale_factor, min_neighbors, resize_factor, save_video, filename)`: Detects full bodies in a video stream.

## Detailed Explanation of the Functions
### 1. `detect_face(image_path, save_fig=False, filename=None)`

This function detects faces in a given image using Haar Cascade classifiers.

![katherine_face](https://github.com/user-attachments/assets/61179f83-d74d-48e5-8978-6ea6d77de0e2)

#### Parameters:
- `image_path`: String, path to the input image file.
- `save_fig`: Boolean, whether to save the resulting figure (default: False).
- `filename`: String, file name to save the figure (if save_fig is True).

#### Functionality:
1. Loads the pre-trained face cascade classifier.
2. Reads the input image and converts it to grayscale.
3. Detects faces using the `detectMultiScale` method of the face cascade.
4. Draws rectangles around detected faces.
5. Displays the original image and the image with detected faces side by side.
6. Optionally saves the resulting figure.

#### Key Features:
- Uses `cv2.CascadeClassifier` with 'haarcascade_frontalface_default.xml'.
- Implements error handling for file loading and face detection.
- Utilizes matplotlib for image display and saving.

### 2. `detect_face_and_eye(image_path, save_fig=False, filename=None)`

This function detects both faces and eyes in a given image.

![katherine_eye](https://github.com/user-attachments/assets/0e51b82e-de8a-4442-8b3f-89f00d2ae77e)

![ronaldo_eyes](https://github.com/user-attachments/assets/0248bbb2-1f4b-4744-b742-b314b2156c3a)

![group_selfie_eyes](https://github.com/user-attachments/assets/b73f9de3-e701-4155-83c5-a8f495ef29a9)


#### Parameters:
- Same as `detect_face` function.

#### Functionality:
1. Loads both face and eye cascade classifiers.
2. Detects faces in the image.
3. For each detected face, it creates a region of interest (ROI) and detects eyes within this region.
4. Draws rounded rectangles around faces and rectangles around eyes.
5. Displays and optionally saves the result.

#### Key Features:
- Uses both face and eye Haar Cascades.
- Implements nested detection (eyes within faces).
- Uses custom `draw_rounded_rect` function for aesthetically pleasing face bounding boxes.

In some cases, it may mistakenly identify other facial features as eyes in the image. To ensure that we only detect and draw bounding boxes around the eyes, and not other facial regions such as lips or eyebrows, we can filter the detected eye regions based on their relative vertical positions. Specifically, if the centres of two detected eye-bounding boxes are within a specified vertical range (e.g., ±50 pixels), we can confirm that they are likely to be eyes.

![katherine_eye_1](https://github.com/user-attachments/assets/58e87210-5d64-43a4-9e84-56c12511e7a4)

![sarapova_eyes](https://github.com/user-attachments/assets/6c2ecffa-0410-424e-9b90-bff000664667)

### 3. `detect_human_body(image_path, scale_factor, min_neighbours, save_fig=False, filename=None)`

This function detects full human bodies in an image.

![body_detection_2](https://github.com/user-attachments/assets/eb650e6b-e19c-41e0-9331-2d8e8f1e7525)

As seen in the figure, the haar cascade is less accurate in detecting persons compared to other object detection models (e.g., YOLO, detectron2). It misses out on many persons and sometimes doesn't capture the whole human body.
#### Parameters:
- `image_path`: String, path to the input image (can be a URL or local path).
- `scale_factor`: Float, parameter specifying how much the image size is reduced at each image scale.
- `min_neighbours`: Integer, parameter specifying how many neighbors each candidate rectangle should have to retain it.
- `save_fig` and `filename`: Same as previous functions.

#### Functionality:
1. Loads the full-body Haar Cascade classifier.
2. Handles both local and URL-based image inputs.
3. Detects full bodies in the image.
4. Draws rectangles and labels for each detected body.
5. Displays and optionally saves the result.

#### Key Features:
- Uses 'haarcascade_fullbody.xml' for detection.
- Handles URL-based image inputs using the `requests` library.
- Implements custom text drawing with background for clear labeling.

### 4. `detect_faces_from_video(video_path, scale_factor, min_neighbors, resize_factor, save_video=False, filename=None)`

This function detects faces in a video stream.

![face_detection_1](https://github.com/user-attachments/assets/79ceae57-cdaf-4817-9001-382121bd5c93)

#### Parameters:
- `video_path`: String, path to the input video file or camera index.
- `scale_factor` and `min_neighbors`: Same as in `detect_human_body`.
- `resize_factor`: Float, factor to resize the displayed video frames.
- `save_video`: Boolean, whether to save the processed video.
- `filename`: String, file name to save the processed video.

#### Functionality:
1. Opens the video stream and prepares video writer if saving is requested.
2. Processes each frame of the video:
   - Detects faces
   - Draws rounded rectangles around faces
   - Detects eyes within face regions for additional validation
3. Displays the processed frames in real-time.
4. Optionally saves the processed video.

#### Key Features:
- Real-time face detection in video streams.
- Uses eye detection to validate face regions.
- Implements video saving functionality.

### 5. `detect_human_body_from_video(video_path, scale_factor, min_neighbors, resize_factor, save_video=False, filename=None)`

This function detects full human bodies in a video stream.

![body_detection_vid](https://github.com/user-attachments/assets/b9a1f221-6797-4a56-8d65-41d31f58acc3)

Here, the number associated with the body is not the tracker ID; it is only used to display the number of bodies the model detected.

#### Parameters:
- Same as `detect_faces_from_video` function.

#### Functionality:
1. Opens the video stream and prepares video writer if saving is requested.
2. Processes each frame of the video:
   - Detects full bodies
   - Draws rectangles around detected bodies
   - Labels each detected body
3. Displays the processed frames in real-time.
4. Optionally saves the processed video.

#### Key Features:
- Real-time full-body detection in video streams.
- Implements labeling for multiple detected bodies.
- Provides options for video resizing and saving.

These functions collectively provide a comprehensive toolkit for detecting human features (faces, eyes, and full bodies) in images and videos using Haar Cascade classifiers. They offer flexibility in input handling, parameter tuning, and output options, making them suitable for various computer vision applications.


