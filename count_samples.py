import pandas as pd
import cv2
import os
import glob
import numpy as np



# df = pd.read_csv('dataset_and_augmented.csv', index_col= None)
df = pd.read_csv('dataset_2_.csv', index_col= None)
df = df.set_index('img')
df = df.replace(np.nan, 0)

df = df.drop(columns=['crooked', 'missing', '52-Crooked', 'chopped'])


def count_samples(images_path):
    my_samples = np.zeros((1,7))
    for f in glob.glob(os.path.join(images_path, "*")):

        name = os.path.splitext(os.path.basename(f))[0]
        extension = os.path.splitext(os.path.basename(f))[1]

        if (name+extension in df.index):
            filename = name+extension
            labels = df.loc[filename].values
            my_samples += labels
    return my_samples

path = './all_teeth/teeth_padded/'

samples = np.zeros((4, 7))
samples[0] = count_samples(path+"train")
samples[1] = count_samples(path+"valid")
samples[2] = count_samples(path+"test")
samples[3] = samples[0] + samples [1] + samples [2]
# print(samples)


samples_df = pd.DataFrame(samples,index=['train', 'valid', 'test', 'total'], columns=df.columns)
print(samples_df)
