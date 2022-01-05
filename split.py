import pandas as pd
import glob
import os
import numpy as np
from csv import writer


df = pd.read_csv('dataset_2_.csv', index_col= None)
df = df.set_index('img')
df = df.replace(np.nan, 0)

df = df.drop(columns=['crooked', 'missing', '52-Crooked', 'chopped'])



images_path = "./all_teeth/teeth_padded/test"

for f in glob.glob(os.path.join(images_path, "*")):

    name = os.path.splitext(os.path.basename(f))[0]
    extension = os.path.splitext(os.path.basename(f))[1]

    if (name+extension in df.index):
        filename = name+extension
        labels = df.loc[filename].values

    list_data = []
    list_data.append(filename)
    for i in labels:
        list_data.append(i)

    with open('test.csv', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        
        writer_object.writerow(list_data)  
        f_object.close()
    