import numpy as np
import cv2
import os
import glob


def resize_image(img, size=(224, 224)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


images_path = "./all_teeth/teeth/augmented/valid"
out_path = "./all_teeth/teeth_padded/valid"

# images_path = "./cropped_teeth"
# out_path = "./resized_cropped_teeth"

for f in glob.glob(os.path.join(images_path, "*")):
    # print("Processing file: {}".format(f))
    img = cv2.imread(f)
    resized_img = resize_image(img)

    filename = os.path.splitext(os.path.basename(f))[0]
    extension = os.path.splitext(os.path.basename(f))[1]
    # print(filename + extension)

    cv2.imwrite(out_path+"/"+filename+extension,resized_img)
