#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:22:52 2023

@author: rafael
"""

import cv2 
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Function to check if sign is detected based on the fingers that are ope n
def peace(fingers_open,tumb_between):
    # Extra condition for location of thumb
  if fingers_open == [1,1,1,0,0] and tumb_between == True:
    return True
  else:
    return False

def surf(fingers_open):
  if fingers_open == [1,0,0,0,1]:
    return True
  else: 
    return False

def hand_open(fingers_open):
  if fingers_open == [1,1,1,1,1]:
    return True
  else: 
    return False

# Function to compute distance between two points
def dist_points(l,i1,i2):
  dist = (((l[i1].x - l[i2].x)**2) + (l[i1].y - l[i2].y)**2) **0.5
  return dist


def fingers_configuration(l) :
    hand_fingers = []
    for i in range(4,24,4): # iterate over landmarks on tip of fingers
    
    # Compute distance of Tip and MCP to the Wrist
      dist_tip =  dist_points(l,i,0) 
      dist_mcp = dist_points(l,i-2,0)
      if dist_tip > dist_mcp:
      
        hand_fingers.append(1)
      else:
        hand_fingers.append(0)
        
    return hand_fingers

def get_signs(detection_result):
  hand_landmarks_list = detection_result.hand_landmarks # get landmarks from results
  signs = []
  
  for l in hand_landmarks_list:
    if dist_points(l,4,9) < dist_points(l,4,6): # Compute distance between tip thumb and (middle_MCP and index_PIP)
      tb2 = True
    else :
      tb2 = False

    hand_fingers = fingers_configuration(l)

    #Store the found signs
    if peace(hand_fingers,tb2):
        signs.append('PEACE')
    elif surf(hand_fingers):
        signs.append('SURF')
    elif hand_open(hand_fingers):
        signs.append('HAND OPEN')
    else:
        signs.append(' ') # If hand detected but no sign
    
  return signs

# Get connections between landmarks for a specific sign based on the fingers that are open
# CMC should be given, then all others landmarks are found for that finger
def pos_connections(i, hand_landmarks, frame):
    pos_lm = [(int(hand_landmarks[i].x * frame.shape[1]), int(hand_landmarks[i].y * frame.shape[0])) for i in range(i, i + 4)]
    connections = [[(int(hand_landmarks[i].x * frame.shape[1]), int(hand_landmarks[i].y * frame.shape[0])),
                    (int(hand_landmarks[i + 1].x * frame.shape[1]), int(hand_landmarks[i + 1].y * frame.shape[0]))]
                   for i in range(i, i + 3)]
    return pos_lm, connections

# Draw landmarks and connections 
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
    
# Compute the the landmarks positions for each sign
def draw_sign(image,signs,detection_result):
    hand_landmarks_list = detection_result.hand_landmarks

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        sign = signs[idx]
        
        if sign == 'PEACE':
            
            pos_idx , con_idx = pos_connections(5,hand_landmarks,image)
            pos_middle , con_middle = pos_connections(9,hand_landmarks,image)
            
            draw_points(image,[pos_idx,pos_middle])
            draw_connections(image,[con_idx,con_middle])
            
        if sign == 'HAND OPEN':
             
            pos_thu , con_thu = pos_connections(1,hand_landmarks,image) 
            pos_idx , con_idx = pos_connections(5,hand_landmarks,image)
            pos_ring , con_ring = pos_connections(13,hand_landmarks,image)
            pos_pin , con_pin = pos_connections(17,hand_landmarks,image)
            pos_middle , con_middle = pos_connections(9,hand_landmarks,image)
            
            draw_points(image,[pos_idx,pos_middle,pos_thu,pos_ring,pos_pin])
            
            draw_connections(image,[con_idx,con_middle,con_thu,con_ring,con_pin])
            
        if sign == 'SURF':
            
            pos_thu , con_thu = pos_connections(1,hand_landmarks,image) 
            pos_pin , con_pin = pos_connections(17,hand_landmarks,image)
            
            draw_points(image,[pos_thu,pos_pin])
            
            draw_connections(image,[con_thu,con_pin])
            
        else:
            None
            
    return 
  
# Get the top left corner of the detected hand's bounding box.
def bounding_corner(annotated_image,hand_landmarks):
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - 10
    return text_x,text_y

def sign(image,detection_result,signs):

    hand_landmarks_list = detection_result.hand_landmarks

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
    
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        # Write sign on the image.
        cv2.putText(image, f"{signs[idx]}",
                    (text_x + 200, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return image

# In macOS still a '.DS_Store' file, thus remove it
def folder_empty(path):
    folder = [image for image in os.listdir(path) if image != '.DS_Store']  
    return len(folder) == 0, folder

# Constants for the visualization of text on the video stream
FONT_SIZE = 2
FONT_THICKNESS = 3
TEXT_COLOR = (255, 0, 0)

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task') # Import trained model from MediaPipe
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

# Paths to images folders
path_images = './images'
new_folder = './images_sign'

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

if not os.path.exists(path_images):
    os.makedirs(path_images)

empty, files = folder_empty(path_images)

# If image folder is empty stream webcam otherwise process images for signs and store them in a new folder
if not empty:    
    for file in files:
        image_path = os.path.join(path_images,file)
           
        # Load the input image from a numpy array to a MediaPipe's Image object
        image = cv2.imread(image_path)
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image to RGB (MediaPipe requirement)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im_rgb)
    
        # Detect hand landmarks from the input image
        detection_result = detector.detect(mp_image)
        
        #Process the result. Get signs if present and visualize them
        signs = get_signs(detection_result)
                        
        draw_sign(image,signs,detection_result)
        sign(image,detection_result,signs)
        
        # Display the resultling frame
        cv2.imwrite(os.path.join(new_folder,file),image)
    
else:
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
        
        if len(detection_result.hand_landmarks) > 0: # Only do the computation if there is a hand detected
        
            #Process the result. Get signs if present and visualize them
            signs = get_signs(detection_result)
                        
            draw_sign(frame,signs,detection_result)
            sign(frame,detection_result,signs)
        
        # Display the resultling frame
        cv2.imshow('frame',frame)
        
        key = cv2.waitKey(1) 
        if key == ord('q'):        
            cv2.destroyAllWindows() # Destroy windows
            cv2.waitKey(1)   # Otherwise it will not close on macOS
            print('destroy')
            break
    
    cap.release()
