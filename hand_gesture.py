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


# Function to check if sign is detected
def peace(fingers_open,tumb_between):
  if fingers_open == [1,1,1,0,0] and tumb_between == True:
    return True
  else:
    return False

def surf(fingers_open):
  if fingers_open == [1,0,0,0,1]:
    return True
  else: 
    return False

def hand_open(fingers_open,tb):
  if fingers_open == [1,1,1,1,1] :#and tb == False:
    return True
  else: 
    return False

# Function to compute distance between two points
def dist_points(l,i1,i2):
  dist = (((l[i1].x - l[i2].x)**2) + (l[i1].y - l[i2].y)**2) **0.5
  return dist


def fingers_configuration(l) :
    hand_fingers = []
    for i in range(4,24,4): # landmarks on tip of finger
    
    # Compute distance of Tip and MCP to the Wrist
      dist_tip =  dist_points(l,i,0) 
      dist_mcp = dist_points(l,i-2,0)
      if dist_tip > dist_mcp:
      
        hand_fingers.append(1)
      else:
        hand_fingers.append(0)
        
    return hand_fingers

def get_gestures(detection_result):
  Landmarks = detection_result.hand_landmarks # get landmarks from results
  Hands = detection_result.handedness # get Handedness 
  
  for l , h in zip(Landmarks,Hands):
    

    if dist_points(l,4,13) < dist_points(l,4,5): #13 of 9 
      tb2 = True
    else :
      tb2 = False

    hand_fingers = fingers_configuration(l)
        
    if peace(hand_fingers,tb2):
        gestures.append('PEACE')
    elif surf(hand_fingers):
        gestures.append('SURF')
    elif hand_open(hand_fingers,tb2):
        gestures.append('HAND OPEN')
    else:
        gestures.append(' ')    # If hand detected but no gesture
    
  return gestures

def pos_connections(i, hand_landmarks, frame):
    pos_lm = [(int(hand_landmarks[i].x * frame.shape[1]), int(hand_landmarks[i].y * frame.shape[0])) for i in range(i, i + 4)]
    connections = [[(int(hand_landmarks[i].x * frame.shape[1]), int(hand_landmarks[i].y * frame.shape[0])),
                    (int(hand_landmarks[i + 1].x * frame.shape[1]), int(hand_landmarks[i + 1].y * frame.shape[0]))]
                   for i in range(i, i + 3)]
    return pos_lm, connections

def draw_points(image, positions):
    for position in positions:
        for c in position:
            cv2.circle(img=image, center=c, radius=3, color=(255, 0, 0), thickness=2)
    return image

def draw_connections(image,connections):
    for connection_list in connections:
        for line in connection_list:
            
            cv2.line(image,line[1],line[0],color=(255, 0, 0), thickness=2)
    return image
    
    
def draw_gesture(image,gestures,detection_result):
    hand_landmarks_list = detection_result.hand_landmarks

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        gesture = gestures[idx]
        
        if gesture == 'PEACE':
            
            pos_idx , con_idx = pos_connections(5,hand_landmarks,image)
            pos_middle , con_middle = pos_connections(9,hand_landmarks,image)
            
            draw_points(image,[pos_idx,pos_middle])
            draw_connections(image,[con_idx,con_middle])
            
        if gesture == 'HAND OPEN':
             
            pos_thu , con_thu = pos_connections(1,hand_landmarks,image) ### VERANDER NAMEN
            pos_idx , con_idx = pos_connections(5,hand_landmarks,image)
            pos_ring , con_ring = pos_connections(13,hand_landmarks,image)
            pos_pin , con_pin = pos_connections(17,hand_landmarks,image)
            pos_middle , con_middle = pos_connections(9,hand_landmarks,image)
            
            draw_points(image,[pos_idx,pos_middle,pos_thu,pos_ring,pos_pin])
            
            draw_connections(image,[con_idx,con_middle,con_thu,con_ring,con_pin])
            
        if gesture == 'SURF':
            
            pos_thu , con_thu = pos_connections(1,hand_landmarks,image) ### VERANDER NAMEN
            pos_pin , con_pin = pos_connections(17,hand_landmarks,image)
            
            draw_points(image,[pos_thu,pos_pin])
            
            draw_connections(image,[con_thu,con_pin])
            
        else:
            None
            
    return 

def bounding_corner(annotated_image,hand_landmarks):
    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN
    return text_x,text_y

def gesture(image,detection_result,gestures):

    hand_landmarks_list = detection_result.hand_landmarks

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
    
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Write gesture on the image.
        cv2.putText(image, f"{gestures[idx]}",
                    (text_x + 200, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return image

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
    ret, frame = cap.read() # Capture video frame 
    if not ret:
        break
    
    # Load the input image from a numpy array to a MediaPipe's Image object
    im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert image to RGB (MediaPipe requirement)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im_rgb)

    # Detect hand landmarks from the input image
    detection_result = detector.detect(mp_image)
    
    #Process the result. Get gestures if present and visualize them
    gestures = get_gestures(detection_result)
    
    #annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    
    draw_gesture(frame,gestures,detection_result)
    gesture(frame,detection_result,gestures)

    
    # Display the resultling frame
    cv2.imshow('frame',frame)

    
    key = cv2.waitKey(1) 
    if key == ord('q'):        
        cv2.destroyAllWindows() # Destroy windows
        cv2.waitKey(1)   # Otherwise it will not close on macOS
        print('destroy')
        break

cap.release()
