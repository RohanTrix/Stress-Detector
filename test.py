from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
# loading files
haar_file="models/haarcascade_frontalface_default.xml"
emotion_model='models/_mini_XCEPTION.102-0.66.hdf5'

cascade=cv2.CascadeClassifier(haar_file)
emotion_classifier=load_model(emotion_model,compile=True)
emotion_names=["angry","disgust","scared", "happy", "sad", "surprised",
 "neutral"]

frame=cv2.imread('img1.jpeg')
gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=cascade.detectMultiScale(gray_frame,1.5,5)
text=[]
for (x,y,w,h) in faces:
    roi=gray_frame[y:y+h,x:x+w]
    roi=cv2.resize(roi,(64,64))
    roi=roi.astype("float")/255.0
    roi=img_to_array(roi)
    roi=np.expand_dims(roi,axis=0)
    
    predicted_emotion=emotion_classifier.predict(roi)[0]
    probab=np.max(predicted_emotion)
    label=emotion_names[predicted_emotion.argmax()]
    percen=predicted_emotion*100
    for j in range(7):
        text.append(emotion_names[j]+" : "+str(percen[j]))
    for i in range(7):    
        cv2.putText(frame,text[i],(5,i*15+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imwrite('results/res1.jpg', frame)
