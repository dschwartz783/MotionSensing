from numpy import zeros, add, subtract, sum
import cv2
from time import sleep
from nupic.frameworks.opf.modelfactory import ModelFactory
from model_0 import model_params
import os

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
FRAME_HEIGHT = 256

video_model = ModelFactory.create(model_params.MODEL_PARAMS)

cap = cv2.VideoCapture("http://192.168.1.8/live")

_, frame1 = cap.read()
frame1 = cv2.resize(frame1,
                    (int(round((FRAME_HEIGHT/cap.get(CV_CAP_PROP_FRAME_HEIGHT)*cap.get(CV_CAP_PROP_FRAME_WIDTH)))),
                     FRAME_HEIGHT))
frame1_height, frame1_width, frame1_channels = frame1.shape


initial_delta = zeros((frame1_height, frame1_width, 3))
deltas = [sum(initial_delta)]
video_model.enableInference({"predictedField": "camera_data"})
i = 0
num_infamiliar = 0
while True:
    _, frame2 = cap.read()
    if _:
        frame2 = cv2.resize(frame2, (frame1_width, frame1_height))

        i += 1
        result = initial_delta
        subtract(frame1, frame2, result)
        model_input = {"camera_data": int(sum(abs(result)))}
        output = video_model.run(model_input)
        anomalyScore = output.inferences["anomalyScore"]
        if anomalyScore > 0.4:
            num_infamiliar += 15
        else:
            num_infamiliar -= 1
        if num_infamiliar >= 90:
            print "odd: " + str(i)
            num_infamiliar = 0
        frame1 = frame2
    else:
        break
    if i == 30000:
        video_model.disableLearning()

cap.release()
cv2.destroyAllWindows()
