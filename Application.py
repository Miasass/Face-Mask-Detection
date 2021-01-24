# -*- coding: utf-8 -*-
import cv2
import numpy as np
import keyboard
import imutils
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def Application(face_detecting_file,best_weights):
    face_cascade = cv2.CascadeClassifier(face_detecting_file)

    model = Sequential()

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=3, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights(best_weights)


    cap = cv2.VideoCapture(0)

    while(True):
    
        ret, img = cap.read()
    
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(50,50))
      
        for (x,y,w,h) in faces:
        
            part_of_img = img[y:y+h,x:x+w]
            part_of_img_resized = cv2.resize(part_of_img, (50,50), interpolation = cv2.INTER_AREA)
            np_image_data = np.asarray(part_of_img_resized)
            np_final = np.expand_dims(np_image_data,axis=0)
            np_final = np_final * 1./255.
            y_prob = model.predict(np_final)
            y_pred = np.argmax(y_prob, axis=1)
        
            if y_pred == 0:
            
                cv2.rectangle(img,(x,y),(x+w,y+h),(255, 145, 0),2)
                cv2.putText(img,text='Mask_Weared_Incorrect', org=(100, 50), 
                           	    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       	        color=(255, 145, 0), thickness=1)
            
            elif y_pred == 1:
            
                cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),2)
                cv2.putText(img,text='MASK_WEARED', org=(100, 50), 
                       	       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       	       color=(0, 255, 0), thickness=1)
            
            elif y_pred == 2:
            
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,text='MASK_NOT_WEARED', org=(100, 50), 
                       	       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       	       color=(255, 0, 0), thickness=1)
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imshow('Wykrywanie twarzy',img)
    
        if cv2.waitKey(1) and keyboard.is_pressed('ESC'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    
def main():
    Application('haarcascade_frontalface_alt2.xml','best_model_weights.hdf5')
    return 0

main()

