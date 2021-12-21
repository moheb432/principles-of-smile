import pandas as pd
import cv2
import glob
import os
import re
from csv import writer


df = pd.read_csv('dataset_and_augmented.csv', index_col= None)
df = df.set_index('img')


regExp = '\(([^)]+)\)'

# images_path = "./all_teeth/teeth/augmented/train2"
images_path = "./all_teeth/teeth/augmented/test2"
# images_path = "./all_teeth/teeth/augmented/valid2"
for f in glob.glob(os.path.join(images_path, "*")):
    # print("Processing file: {}".format(f))
    img = cv2.imread(f)

    filename = os.path.splitext(os.path.basename(f))[0]
    real_img_name = re.findall(regExp, filename)
    extension = os.path.splitext(os.path.basename(f))[1]
        
    list_data = []
    list_data.append(filename+extension)
    labels = df.loc[str(real_img_name[0])+extension].values
    for i in labels:
        list_data.append(i)
    
    with open('aug.csv', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        
        writer_object.writerow(list_data)  
        f_object.close()