# -*- coding: utf-8 -*-
"""Instagram scrapper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WBNDgB8ZnsDn8IMLoHJjLElN_l9U5QFJ
"""

from google.colab import drive
drive.mount('/content/drive')

!pip3 install instaloader

#targets = ['amitshahofficial','uddhavthackeray','myogi_adityanath','arvindkejriwal','rahulgandhi','virat.kohli','akshaykumar','ranveersingh','iamsrk','deepikapadukone','katrinakaif','aliaabhatt','priyankachopra','shraddhakapoor','anushkasharma','jacquelinef143','nehakakkar','beingsalmankhan','ratantata'
#,'smritiiraniofficial','srisriravishankar']

targets = ['ranveersingh','deepikapadukone','katrinakaif','priyankachopra','shraddhakapoor','jacquelinef143','nehakakkar','beingsalmankhan','ratantata'
,'smritiiraniofficial','srisriravishankar']
users = ["INSERT USERNAME"]
passwords = ["INSTAGRAM PASSWORD"]
for target in targets:
  !instaloader --login=maskedip1 --password=A123456! --post-filter="date_utc > datetime(2019, 6, 30)" {target} -V --dirname-pattern=/instaloader/india/{target}


