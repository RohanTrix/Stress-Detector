from keras.preprocessing.image import img_to_array
import cv2
from flask import Flask, render_template, Response
from keras.models import load_model
import numpy as np
# loading files
haar_file="models/haarcascade_frontalface_default.xml"
emotion_model='models/trained_model.hdf5'

# Code inserted for File Upload and Analysis

app = Flask(__name__)

@app.route('/img_upload')
def index():
    """Video streaming home page."""
    return render_template('uploadImage.html')

@app.route('/img_analysis')
def ind():
    """Video streaming home page."""
    return render_template('checkImage.html')

def gen():
    """Video streaming generator function."""

    img = cv2.imread("images/img1.jpeg")
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    frame = cv2.imencode('.jpg', img)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Above code is inserted for File Upload and Analysis

cascade=cv2.CascadeClassifier(haar_file)
emotion_classifier=load_model(emotion_model,compile=True)
emotion_names=["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
frame=cv2.imread('images/img1.jpeg')
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
        cv2.putText(frame,text[i],(5,i*30+15),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        print(text[i])
    cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imwrite('results/res1.jpg', frame)


if __name__ == '__main__':
    app.run(debug=True)
