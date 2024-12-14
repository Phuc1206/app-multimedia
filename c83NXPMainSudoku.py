# real_time_sudoku.py
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import realtimesudoku  # Ensure you have the realtimesudoku module

def run_real_time_sudoku(video_path=None):
    # Load video or open camera
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1280)  # Width
        cap.set(4, 720)   # Height

    input_shape = (28, 28, 1)
    num_classes = 9

    # Load model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.load_weights("./C087NXuanPhucModel/digitRecognition.h5")  # Load weights

    old_sudoku = None

    while True:
        ret, frame = cap.read()
        if ret:
            sudoku_frame = realtimesudoku.recognize_and_solve_sudoku(frame, model, old_sudoku)
            cv2.imshow("Real Time Sudoku Solver", cv2.resize(sudoku_frame, (1066, 600)))

            # Exit conditions
            if cv2.getWindowProperty("Real Time Sudoku Solver", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break  # End of video or error in reading frame

    cap.release()
    cv2.destroyAllWindows()
