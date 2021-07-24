# MaskTrendExtension
It is an analysis based project where we analyse famous celebrities mask wearing trend in USA, Brazil and India and their impact on COVID-19 cases
Steps to use:
1.Scrapping:Run instagram_scrapper.py with your instagram id to scrap data from instagram.It uses instaloader API to collect data from instagram. Read more about instaloader here https://instaloader.github.io/cli-options.html
2.Face Extraction: Run maskextension.ipynb Jupyternotebook for eaxtracting face region and saving it into face_data.pkl(given in repo), it takes instaloader dataset and scans images and saves face region coordinates as dictionary for each image.
3.Jaw Region Extraction: Run mask_region.py on images to extract and save jaw region for each faces.It creates a directory mouths containing monthwise jaw region for various celebrirties of different countries.
4.Mask Detection: For mask detection run inference.py, it crates a dictionary and save as inference.py.This dictionary contains 0 when mask is applied and 1 when mask is not applied.
5.Analysis: Analysis is done in maskextension.ipynb notebook.(in last three cells.).

Different files and their purpose:

imagepath.pkl : pickle file contains path of each image in dataset.
face_data.pkl : pickle file containing bounding box cordinates for each face in the posts.
mask_classifier.h5 : model weight for mask classification
inference.pkl: Contains results whether mask is detected for a jaw region or not.
