from numpy import zeros, add, subtract, sum
import cv2
from time import sleep
from csv import writer, reader

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
FRAME_HEIGHT = 256

csv_file = open("video_fingerprint.csv", "wb")

csv_writer = writer(csv_file, delimiter=",")
csv_writer.writerow(["camera_data"])
csv_writer.writerow(["int"])
csv_writer.writerow([""])

cap = cv2.VideoCapture("http://192.168.1.8/live")

_, frame1 = cap.read()
frame1 = cv2.resize(frame1,
                    (int(round((FRAME_HEIGHT/cap.get(CV_CAP_PROP_FRAME_HEIGHT)*cap.get(CV_CAP_PROP_FRAME_WIDTH)))),
                     FRAME_HEIGHT))
frame1_height, frame1_width, frame1_channels = frame1.shape


initial_delta = zeros((frame1_height, frame1_width, 3))
deltas = [sum(initial_delta)]

while True:
    _, frame2 = cap.read()
    if _:
        frame2 = cv2.resize(frame2, (frame1_width, frame1_height))

        result = initial_delta
        subtract(frame1, frame2, result)
        csv_writer.writerow([int(sum(abs(result)))])
        frame1 = frame2
    else:
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
