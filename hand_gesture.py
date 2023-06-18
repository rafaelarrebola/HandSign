#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:22:52 2023

@author: rafael
"""

import numpy as np
import cv2 #as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# From : https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=SE6_sPCXaX3g
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# Constants for the visualization of text on the video stream
MARGIN = 10  # pixels
FONT_SIZE = 1.5
FONT_THICKNESS = 3
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#### werk met if landmarks is not none

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task') # Import trained model from MediaPipe
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
# change to video ? only has reduction in a computational way
# change number of hands?

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0) # Video capture object for Webcam

if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = 0
fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS of Webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Load the input image from a numpy array to a MediaPipe's Image object
    im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert image to RGB (MediaPipe requirement)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im_rgb)

    # Detect hand landmarks from the input image
    detection_result = detector.detect(mp_image)
    
    #Process the result. Visualize them
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        
    # only visualize the sign!
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow('frame',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows() # Destroy windows
        cv2.waitKey(1)   # Otherwise it will not close on macOS
        print('destroy')
        break

cap.release()
