from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import sys


def classify(modelPath,labelPath,count):
        if(count==0):
                modelPath = input("Where is the model?")
                labelPath = input("Where are the labels?")
                testImage = input("What image do you want to classify?")
        else:
                testImage = input("What image do you want to classify?")
        image = cv2.imread(testImage)
        output = image.copy()
        image = cv2.resize(image, (200, 200))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        print("[INFO] loading network...")
        model = load_model(modelPath)
        lb = pickle.loads(open(labelPath, "rb").read())

        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx-1]

        filename = testImage.split("/")[-2]
        correct = "correct" if filename.rfind(label) != -1 else "incorrect"

        # build the label and draw the label on the image
        label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
        output = imutils.resize(output, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        if(correct=="incorrect"):
           print("[REAL] {}: {:.2f}%".format(filename,proba[int(filename)]*100))
        # show the output image
        print("[INFO] {}".format(label))
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        cont = input("Do you have another image? y/n ")
        if(cont!='n'):
                classify(modelPath,labelPath,count+1)
        else:
                sys.exit(1)

classify('','',0)
