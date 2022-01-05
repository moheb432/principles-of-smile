# Principles of Smile Design - Deep Learning Approach
This is a project for Biometrics course -SBE462- tackling the [21 principles of smile design](https://centerfordentalhealth.com/wp-content/uploads/2019/01/21-Principles-of-Smile-Design.pdf) as a multi-label classification problem using Deep Learning.


## _[DATASET](https://drive.google.com/drive/folders/15dE0dFRPq7mTZ8c-zVIag1TsFryztLD0)_
-----------------

![Build Status](https://www.greatdentalexpressions.com/blog/wp-content/uploads/2018/01/BP-diverse-smile.jpeg)

A total of **682** images of smiley faces and teeth -before and after- were collected belonging to **7** classes:
- Gummy Smile
- Incisal Embrasure
- Color
- Central Incisor W/H Ratio
- Black Triangle
- Gaps
- Crooked Teeth

After collecting at least **30** images for each class, the data was labeled according to the [dental_AI.ppt](https://docs.google.com/presentation/d/1GACdmOhz4q5GXNtHChhvm9ulD4B5ZWgA/edit#slide=id.p1) presenetation descibing the aesthetic issues for the mentioned 7 classes. ([csv link](https://docs.google.com/spreadsheets/d/1YaPpBtqj4uY9rEL9IHrajS_NhcxVBa25J1uie6wgkqc/edit?usp=sharing))



### *Data Augmentation*

We employed the data expansion pre-processing approach as the training data was not enough for the 7 classes, to increase the robustness of the model. The total resulting images were about **2500**. The augmentation techniques were as follows:
``` 
datagen = ImageDataGenerator(        
            rotation_range=30,      
            width_shift_range=0.2,  
            height_shift_range=0.2,    
            zoom_range=0.2,        
            horizontal_flip=True,
  ```

![Augmentation](documentation/aug.png =100x100)
<img src="documentation/aug.png" width="500" height="500">

## VGG-16 Model with Transfer Learning
-----------------
We used the same architecture as the VGG-16 baseline model + transfer learning by loading pretrained weights from a face recognition task and fine-tuning.

Added regularization and dropout layers in the classification layers of the model. We then trained the model for three ratios (*60:15:25, 70:10:20 and 80:5:15*) of train, valid and test datasets and performed K-fold cross validation to ensure generalization.
# GUI
- the gui is made to apply prediction on a special testing set of our teeths with our names
- it also previews the image and the predicted problems

![Build Status](https://github.com/moheb432/principles-of-smile/blob/main/.idea/2.PNG)

![Build Status](https://github.com/moheb432/principles-of-smile/blob/main/.idea/Capture.PNG)

![Build Status](https://github.com/moheb432/principles-of-smile/blob/main/.idea/3.PNG)
