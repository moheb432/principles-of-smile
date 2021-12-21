from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os
import glob


datagen = ImageDataGenerator(        
            rotation_range=30,
            width_shift_range=0.2,  
            height_shift_range=0.2,    
            zoom_range=0.2,        
            horizontal_flip=True,         
            fill_mode='constant')


images_path = './all_teeth/teeth/valid2'
out_path = './all_teeth/teeth/augmented/valid2'
for f in glob.glob(os.path.join(images_path, "*")):

    x = io.imread(f)
    x = x.reshape((1, ) + x.shape)


    filename = os.path.splitext(os.path.basename(f))[0]
    extension = os.path.splitext(os.path.basename(f))[1]
    i = 0
    for batch in datagen.flow(x, batch_size=32,
                            save_to_dir=out_path,
                            save_prefix='aug_('+filename+')',
                            save_format= extension[1:]): 
        i += 1    
        if i > 2:        
           break