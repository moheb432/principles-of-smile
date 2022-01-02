from PyQt5.QtWidgets import QApplication,QVBoxLayout,QTableWidget,QTableWidgetItem, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem
from PyQt5.QtGui import  QPixmap,QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
import os
from os import path
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import sys
import seaborn as sns
import numpy as np

THIS_FOLDER= path.dirname(path.abspath(__file__))
MAIN_CLASS,_=loadUiType(path.join(THIS_FOLDER, "main.ui"))

classes1 = ['9-gummy', '16-Incisal Embrasure', '4-color', '41-Central Incisor W/H Ratio','18-black triangle','8-gaps', '52-Crooked']

class Main(QtWidgets.QMainWindow,MAIN_CLASS):
    def __init__(self,parent=None):
        super(Main,self).__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.open.triggered.connect(lambda:self.new())
        self.about.triggered.connect(lambda:self.new())
        self.combo.currentIndexChanged.connect(lambda:self.cb())
        self.model=keras.models.load_model("7classes_b16_lr0002.h5")

    def new(self):
        try:
            loadSignal= QtWidgets.QFileDialog.getOpenFileName( self, 'Open only jpg ', os.getenv('HOME') ,"jpg(*.jpg)")
            path=loadSignal[0]
            scene = QGraphicsScene()
            scene.addPixmap(QPixmap(path))
            self.images.setScene(scene)
            self.images.show()
            image = load_img(path,target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            predictions_7= self.model.predict(image)
            y_pred_7=[]
            for pred in predictions_7:
                pred[pred>0.5]=1
                pred[pred<=0.5]=0
                y_pred_7.append(pred)
            l=[]
            for i in range(7):
                if y_pred_7[0][i]==1 :
                        l.append(classes1[i])
            
            self.resultsTable.setColumnCount(len(l))
            self.resultsTable.setRowCount(1)
            
            for row in range(len(l)):
                self.resultsTable.setItem(0, row, QtWidgets.QTableWidgetItem(str(l[row])))
               
            self.resultsTable.verticalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        except:
            self.resultsTable.clear()
            pass   
    def cb(self):
      try:
            x=self.combo.currentIndex()    
            if( x==0):
                loadSignal= QtWidgets.QFileDialog.getOpenFileName( self, 'Open only jpg ', os.getenv('HOME') ,"jpg(*.jpg)")
                path=loadSignal[0]
            
            if( x==1):path=r"C:\Users\lenovo\Desktop\analysis of smiles\Our_cropped_teeth\Karin.jpg"
            if( x==2):path=r"C:\Users\lenovo\Desktop\analysis of smiles\Our_cropped_teeth\fatma_G.jpg" 
            if( x==3):path=r"C:\Users\lenovo\Desktop\analysis of smiles\Our_cropped_teeth\fatma_O.jpg"   
            if( x==4):path=r"C:\Users\lenovo\Desktop\analysis of smiles\Our_cropped_teeth\moheb.jpg" 
            if( x==5):path=r"C:\Users\lenovo\Desktop\analysis of smiles\Our_cropped_teeth\mariam.jpg"
            if( x==6):path=r"C:\Users\lenovo\Desktop\analysis of smiles\Our_cropped_teeth\toka.jpg"    
                
                
            
            
            scene = QGraphicsScene()
            scene.addPixmap(QPixmap(path))
            self.images.setScene(scene)
            self.images.show()
            image = load_img(path,target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            predictions_7= self.model.predict(image)
            y_pred_7=[]
            for pred in predictions_7:
                pred[pred>0.5]=1
                pred[pred<=0.5]=0
                y_pred_7.append(pred)
            l=[]
            for i in range(7):
                if y_pred_7[0][i]==1 :
                        l.append(classes1[i])
            
            self.resultsTable.setColumnCount(len(l))
            self.resultsTable.setRowCount(1)
            
            for row in range(len(l)):
                self.resultsTable.setItem(0, row, QtWidgets.QTableWidgetItem(str(l[row])))
               
            self.resultsTable.verticalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
           
      except:
            self.resultsTable.clear()
            pass   
        

   

      
def main():
    app = QtWidgets.QApplication(sys.argv)
    window= Main()
    window.show()  
    app.exec_() 


if __name__ == '__main__':
    main()
