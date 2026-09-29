import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import cv2
import pickle
import time 
import numpy as np

BaseOptions = mp.tasks.BaseOptions

HAND_LANDMARKER_PATH = "models/hand_landmarker.task"
MODEL_PATH = "models/own_nn_2.pkl"


letters = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) not in ["J", "Z"]]
idx_2_letter = {idx: label for idx, label in enumerate(letters)}


def get_coords_as_list(hand_landmarks):
    coords = []

    for landmark in hand_landmarks:
        coords.append(landmark.x)
        coords.append(landmark.y)

    return coords

def get_prediction(nn):
    predictions = nn.layers[-1].after_activation 
    return np.argmax(predictions)


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape  

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]   

    for landmark in hand_landmarks:
        x = int(landmark.x * w)   # normalizalva vannak alapbol
        y = int(landmark.y * h)

        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)   # -1 hogy ne csak korvanalat rajzoljon, hanem toltse ki is a teljes kort 


    for start_idx, end_idx in connections:
        x1 = int(hand_landmarks[start_idx].x * w)
        y1 = int(hand_landmarks[start_idx].y * h)
        x2 = int(hand_landmarks[end_idx].x * w)
        y2 = int(hand_landmarks[end_idx].y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 



with open(MODEL_PATH, "rb") as f:
    nn = pickle.load(f) 


    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path = HAND_LANDMARKER_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )


    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)   

        while True:
            _, frame = cap.read()

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
                )

            timestamp_ms = int(time.time() * 1000)
    
            result = landmarker.detect_for_video(image, timestamp_ms)

            if len(result.hand_landmarks) > 0:
                hand_landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, hand_landmarks)

                coords = get_coords_as_list(hand_landmarks)

                nn.feed_forward(np.array(coords))

                prediction = get_prediction(nn)
    
                letter = idx_2_letter[prediction]
                
                cv2.putText(frame, f"Prediction: {letter}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("Camera", frame)
            cv2.waitKey(1)
