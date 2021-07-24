import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from retinaface.pre_trained_models import get_model as get_detector
from tqdm import tqdm
import json
from pathlib import Path
import sys
from glob import glob

import pickle
import joblib
# Saving and Loading models using joblib 
def save(filename, obj):
  with open(filename, 'wb') as handle:
      joblib.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
  with open(filename, 'rb') as handle:
      return joblib.load(filename)


model=load_model('mask_classifier.h5')
inference = {}
countries=glob("mouths/*")
for country in countries:
	print(country)
	inference[country.split("\\")[-1]] = {}
	folders = glob(country+"/*")	
	for folder in folders:
		print(folder)
		images = [i for i in glob(folder+"/*.jpg")]
		loaded_images = []
		for image in images:
			image_name = str(image).split("/")[-1]
			try:
				roi=Image.open(image)
				loaded_images.append(img_to_array(roi.resize((224,224))))
			except:
				continue
		X_test = np.array(loaded_images)
		#print(f"X_test shape = {X_test.shape}")
		predict_idxs = model.predict(X_test, batch_size=64)
		predict_idxs = np.argmax(predict_idxs, axis=1)
		inference[country.split("\\")[-1]][folder.split("\\")[-1]] = predict_idxs

save('inference.pkl',inference)
		#print(predict_idxs) 
'''
		with open(f"/app/data/prediction/{folder}.jsonl","a") as f:
			for i in range(len(batch)):
				image_name = str(batch[i]).split("/")[-1]
				f.write(json.dumps({
					image_name : int(predict_idxs[i])
				}) + "\n")
			
'''