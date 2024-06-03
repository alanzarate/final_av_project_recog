# import os
# import shutil
# import uuid

# uuids = []
# category = []
# types = []

# def copy(src, dst):
#     shutil.copyfile(src, dst)
# #shutil.copyfile(src, dst)

# for root, dirs, files in os.walk("./train"):
    
#     if(len(files) > 0):
#         nameSplitted = root.split("/")
#         for fil in files:
#             src = root + "/"+ fil
#             uudiStr =   str(uuid.uuid4())

#             dst = f"./prodCarlos/{uudiStr}"
#             typeStr = nameSplitted[-1]
#             categoryStr =  nameSplitted[-2]
#             copy(src, dst)
#             uuids.append(uudiStr)
#             category.append(categoryStr)
#             types.append(typeStr)
            
#         print(nameSplitted[-1], nameSplitted[-2] )
#         print(root)
        
#         print(len(files))

import cv2
import numpy as np
import pandas as pd
preDF = pd.read_csv("prod_data_images_v3.csv")
n = int(preDF.shape[0]*0.4)
df = preDF.sample(n = n, random_state=40)
#totalImages = np.empty((n,400,400,3))
totalImages = []

for index, row in df.iterrows():
    print(index)
    uuid = row["UUID"]
    routeImage =  f"./prodFinal/{uuid}.jpg"
    image = cv2.imread(routeImage)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #totalImages = np.concatenate((totalImages, [image_rgb]), axis=0)
    totalImages.append(image_rgb)
totalImages = np.array(totalImages)
np.save('loadedImages.npy', totalImages)  