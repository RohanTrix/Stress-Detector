# from scipy.spatial import distance as dist
# from imutils import face_utils
# import numpy as np
# import math
# import imutils
# import time
# import dlib
# import cv2
# from cv2 import VideoWriter_fourcc, VideoWriter
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import image_dataset_from_directory
# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
from flask import Flask, render_template, request, jsonify, redirect
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/checkstress', methods = ['GET', 'POST'])
def check_stress():
    return render_template('check.html')



# def eye_brow_distance(leye,reye):
#     global points
#     distq = dist.euclidean(leye,reye)
#     points.append(int(distq))
#     return distq

# def emotion_finder(faces,frame):
#     global emotion_classifier
#     EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
#     x,y,w,h = face_utils.rect_to_bb(faces)
#     frame = frame[y:y+h,x:x+w]
#     roi = cv2.resize(frame,(64,64))
#     roi = roi.astype("float") / 255.0
#     roi = img_to_array(roi)
#     roi = np.expand_dims(roi,axis=0)
#     preds = emotion_classifier.predict(roi)[0]
#     emotion_probability = np.max(preds)
#     label = EMOTIONS[preds.argmax()]
#     if label in ['scared','sad']:
#         label = 'stressed'
#     else:
#         label = 'not stressed'
#     return label
    
# def normalize_values(points,disp):
#     normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
#     stress_value = np.exp(-(normalized_value))
#     if stress_value>=65:
#         return stress_value,"High Stress"
#     else:
#         return stress_value,"low_stress"
    
# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# emotion_classifier = load_model("models/_mini_XCEPTION.102-0.66.hdf5", compile=False)
# cap = cv2.VideoCapture('vid2.mp4')
# points = []
# stress_list = []
# stressval_list = []
# stressgraph = []
# size=0
# while(True):
#     _,frame = cap.read()
#     if(not _): break
#     frame = cv2.flip(frame,1)
#     frame = imutils.resize(frame, width=500,height=500)
    
    
#     (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
#     (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

#     #preprocessing the image
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
#     detections = detector(gray,0)
#     for detection in detections:
#         emotion = emotion_finder(detection,gray)
#         cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         shape = predictor(frame,detection)
#         shape = face_utils.shape_to_np(shape)
           
#         leyebrow = shape[lBegin:lEnd]
#         reyebrow = shape[rBegin:rEnd]
            
#         reyebrowhull = cv2.convexHull(reyebrow)
#         leyebrowhull = cv2.convexHull(leyebrow)

#         cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

#         distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
#         stress_value,stress_label = normalize_values(points,distq)
#         print(stress_value)
#         #if stress_value!=1.0: stress_list.append(stress_list)
#         if math.isnan(stress_value):
#             continue
#         cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     height, width, layers = frame.shape
#     size = (width,height)
    
#     stress_list.append(frame)
#     stressval_list.append(stress_value)
#     cv2.imwrite("Frame.jpg", frame)
# out = cv2.VideoWriter('resvid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

# cv2.destroyAllWindows()
# cap.release()
# print("END REACHED")
# for i in range(len(stress_list)):
#     out.write(stress_list[i])


if __name__ == '__main__':
    app.run(debug=True)