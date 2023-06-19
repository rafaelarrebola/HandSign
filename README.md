<img width="454" alt="image" src="https://github.com/rafaelarrebola/HandSign/assets/131361835/ff797303-2bc2-46ef-b3ac-2c7b0472034d"># HandSign

## Description 

It detects hand landmarks using MediaPipe and classifies hand gestures based on the fingers that are open. The recognized signs include "Peace", "Surf" and "Open Hand".

The project includes the following files:
- `hand_sign.py`: The main Python script that performs hand sign recognition on either images or a webcam stream. It uses cv2, mediapipe and os. 
- `README.md`: This file providing instructions and information about the project.
- `hand_landmarker.task`: A model from MediaPipe for hand landmark detection

When there is additional folder with images named 'images', the script will run only on the images. Otherwise the webcam will open and the sign detection is done in real time. Only the hand_sign.py script should be run.

First a HandLandmarker object is initialized using MediaPipe and the 'hand_landmarker.task'. In its options the total number of hands that can be detected is 2. The detection result outputs three arrays:
-	Handedness returns if the detected hand is left or right
-	Landmarks are the 21 hand landmarks. The x and y coordinates are normalized by the image size The z coordinate is the depth of the landmark, with as origin the wrist.
- World coordinates with the origin at the hand’s geometric center

The landmarks are based on the following figure. The origin of the coordinates is in the top left corner. With the y-axis pointing downwards. 

If hands are detected the following process is started: 

1) Fingers are considered open if the distance of the TIP to the WRIST is smaller than the distance of the MCP to the WRIST.

2) The 'Open hand' requires are fingers to be open. The 'Surf' needs the thumb and the pinky to be open. For the 'Peace' sign both the middle and index fingers should be open. Additionaly the tip of the thumb should be closer to the MCP of the middle finger than the PIP of the index finger. This constraint makes sure that the thumb finger is resting on the palm of the hand.

3) The signs are visualized by drawing each landmark from CMC to TIP and their inbetween connections. Furthermore the sign is written above the hand. 


