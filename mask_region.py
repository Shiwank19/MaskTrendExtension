import cv2
import argparse
import os
from imutils import face_utils
import numpy as np
from skimage.util import montage
import imutils
from pathlib import Path
import json
import sys
import dlib
from tqdm import tqdm
import pickle
import joblib
# Saving and Loading models using joblib 
def save(filename, obj):
  with open(filename, 'wb') as handle:
      joblib.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
  with open(filename, 'rb') as handle:
      return joblib.load(filename)
predictor = dlib.shape_predictor("pred/shape_predictor_68_face_landmarks.dat") 
face_region = load('face_data_new.pkl')
image_count = 0
total_images = len(face_region)
for region in face_region:
	path = "../"+region['path']
	splitted_path = region['path'].split("\\")
	country = splitted_path[1]
	filename = splitted_path[-1]
	year_month = filename[0:7]
	user = splitted_path[2]
	count = 0
	cv_image = cv2.imread(path,1)
	for info in region['annot']:
		count+=1
		if(len(info['bbox'])<4):
			continue
		x_min, y_min, x_max, y_max = info['bbox']
		x_min = np.clip(x_min, 0, x_max)
		y_min = np.clip(y_min, 0, y_max)
		shape = predictor(cv_image, dlib.rectangle(x_min,y_min,x_max,y_max)) # facial landmarks
		shape = face_utils.shape_to_np(shape)
		points = np.concatenate((shape[4:13],shape[31:36],shape[48:]))
		(x, y, w, h) = cv2.boundingRect(np.array([points]))
		roi = cv_image[y:y + h, x:x + w]
		#print(roi)
		Path(f"mouths/"+country+"/"+year_month+"/").mkdir(parents=True, exist_ok=True)
		#print("Hello")
		cv2.imwrite(f"mouths/"+country+"/"+year_month+"/"+user+"_"+str(count)+".jpg",roi)
	print("Processing Image: "+str(image_count+1)+"/"+str(total_images))	
	image_count+=1	
#jaw_no = 0
#annotation = annotation[0]
