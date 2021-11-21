import sys
import os
import dlib
import glob
import cv2
import numpy as  np



def show_landmarks(landmarks ,image):
    '''
    Display images with face landmarks 
            Parameters:
                    landmarks : landmarks of input image
                    image: input image
    '''
    landmarks_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
    landmarks = landmarks_np

    # Display the landmarks
    for i, (x, y) in enumerate(landmarks):
    # Draw the circle to mark the keypoint 
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    # Display the image
    cv2.imshow('Landmark Detection', image)
    cv2.waitKey(0) 

#def resize()

def extract_mouth(predictor_path,dataset_path):
    '''
    Extract mouth from faces' images
            Parameters:
                    dataset_path: dataset folder path
                    predictor_path: model path
    '''

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    #win = dlib.image_window()

    for f in glob.glob(os.path.join(dataset_path, "*")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        #img = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)

        faces = detector(img)
        
        for d in (faces):
            print("Left: {} Top: {} Right: {} Bottom: {}".format(
                d.left(), d.top(), d.right(), d.bottom()))
        
            # Get the landmarks for the face in box d.
            landmarks = predictor(img, d)

            # Get the coordinates for the mouth
            # mouth points 48:67
            
            # teeth 
            # xmouthpoints = [landmarks.part(x).x for x in range(60,67)]
            # ymouthpoints = [landmarks.part(x).y for x in range(60,67)]

            # mouth
            xmouthpoints = [landmarks.part(x).x for x in range(48,67)]
            ymouthpoints = [landmarks.part(x).y for x in range(48,67)]
            
            maxx = max(xmouthpoints)
            minx = min(xmouthpoints)
            maxy = max(ymouthpoints)
            miny = min(ymouthpoints) 

            # add padding from both sides
            pad = 0

            # get image name
            filename = os.path.splitext(os.path.basename(f))[0]

            crop_image = img[miny-pad:maxy+pad,minx-pad:maxx+pad]
            #cv2.imshow('mouth',crop_image)
            
            # save cropped images in cropped
            cv2.imwrite("cropped_teeth/"+filename+'.jpg',crop_image)
            
            #show_landmarks(landmarks=landmarks,image=img)

            #cv2.destroyAllWindows()
            #win.add_overlay(shape)

if __name__=='__main__':
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    faces_folder_path = "smile_dataset"
    extract_mouth(predictor_path=predictor_path,dataset_path = faces_folder_path)




