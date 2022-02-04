from fileinput import filename
import os, sys, time, urllib.request, traceback
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *

TITLE = "ObDES: Object Detection and depth EStimation"

res_path = "resources/"
weights_Path = res_path + "yolov3.weights"
config_Path = res_path + "yolov3.cfg"
labels_Path = res_path + "coco.names"
name_of_model = "estimator_model"
zip_model_path = res_path + name_of_model + ".zip"
h5_model_path = res_path + name_of_model + ".h5"

def temporary_downloader(path, link):
    urllib.request.urlretrieve(link, path)

if not os.path.exists(res_path):
    os.mkdir(res_path)

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread. Supported signals are:
    error: tuple (exctype, value, traceback.format_exc() )
    progress: int indicating % progress
    '''
    error = pyqtSignal(tuple)
    progress = pyqtSignal(str)

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1018, 556)
        MainWindow.setStyleSheet("background-color: rgb(223, 225, 199);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.CHECK_BTN = QtWidgets.QPushButton(self.centralwidget)
        self.CHECK_BTN.setGeometry(QtCore.QRect(10, 20, 281, 51))
        self.CHECK_BTN.setStyleSheet("background-color: rgb(201, 173, 167);")
        self.CHECK_BTN.setObjectName("CHECK_BTN")

        self.DL_BTN = QtWidgets.QPushButton(self.centralwidget)
        self.DL_BTN.setGeometry(QtCore.QRect(10, 80, 281, 51))
        self.DL_BTN.setStyleSheet("background-color: rgb(201, 173, 167);")
        self.DL_BTN.setObjectName("DL_BTN")

        self.BRW_BTN = QtWidgets.QPushButton(self.centralwidget)
        self.BRW_BTN.setGeometry(QtCore.QRect(10, 140, 281, 51))
        self.BRW_BTN.setStyleSheet("background-color: rgb(201, 173, 167);")
        self.BRW_BTN.setObjectName("BRW_BTN")

        self.START_BTN = QtWidgets.QPushButton(self.centralwidget)
        self.START_BTN.setGeometry(QtCore.QRect(10, 210, 281, 51))
        self.START_BTN.setStyleSheet("background-color: rgb(0, 120, 0); color: rgb(255, 255, 255);")
        self.START_BTN.setObjectName("START_BTN")

        self.NTF_LBL = QtWidgets.QLabel(self.centralwidget)
        self.NTF_LBL.setGeometry(QtCore.QRect(10, 270, 281, 251))
        self.NTF_LBL.setMinimumSize(QtCore.QSize(270, 240))
        self.NTF_LBL.setStyleSheet("background-color: rgb(34, 34, 59); color: rgb(255, 255, 255);")
        self.NTF_LBL.setFrameShape(QtWidgets.QFrame.Box)
        self.NTF_LBL.setFrameShadow(QtWidgets.QFrame.Raised)
        self.NTF_LBL.setMidLineWidth(3)
        self.NTF_LBL.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.NTF_LBL.setWordWrap(True)
        self.NTF_LBL.setObjectName("NTF_LBL")
        
        self.PH_LBL = QtWidgets.QLabel(self.centralwidget)
        self.PH_LBL.setGeometry(QtCore.QRect(300, 20, 711, 501))
        self.PH_LBL.setStyleSheet("background-color: rgb(231, 237, 255);")
        self.PH_LBL.setFrameShape(QtWidgets.QFrame.Box)
        self.PH_LBL.setScaledContents(True)
        self.PH_LBL.setObjectName("PH_LBL")

        self.threadpool = QThreadPool()

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.CHECK_BTN.clicked.connect(self.check_function)
        self.DL_BTN.clicked.connect(self.dl_function)
        self.BRW_BTN.clicked.connect(self.get_image)
        self.START_BTN.clicked.connect(self.do_the_job)

    def check_function(self):
        miss_list = ""
        if not os.path.isfile(config_Path):
            miss_list += "\n  -YOLO CONFIG FILE < 10KB"
        if not os.path.isfile(labels_Path):
            miss_list += "\n  -YOLO LABELS FILE < 1KB"
        if not os.path.isfile(weights_Path):
            miss_list += "\n  -YOLO W8s FILE ~ 240MB"
        if not os.path.isfile(h5_model_path):
            miss_list += "\n  -DEPTH EST MODEL ~ 400MB"
        if miss_list == "":
            NOTIF = "Everything is ready!\n\nBrowse the picture and Start the process!"
        else:
            NOTIF = "Files to be downloaded:" + miss_list
        self.NTF_LBL.setText(NOTIF)
    
    def downloader(self, progress_callback):
        self.CHECK_BTN.setEnabled(False)
        self.DL_BTN.setEnabled(False)
        self.BRW_BTN.setEnabled(False)
        self.START_BTN.setEnabled(False)
        dl_stat = "|| DL Process ||"
        if not os.path.isfile(config_Path):
            dl_stat += "\nYOLO CONFIG file download started..."
            progress_callback.emit(dl_stat)
            try:
                temporary_downloader(config_Path, "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
                dl_stat += "\n  -YOLO CONFIG file downloaded."
            except:
                dl_stat += "\n  -An exception occurred while downloading YOLO CONFIG file."
        progress_callback.emit(dl_stat)
        if not os.path.isfile(labels_Path):
            dl_stat += "\nYOLO LABELS file download started..."
            progress_callback.emit(dl_stat)
            try:
                temporary_downloader(labels_Path, "https://raw.githubusercontent.com/pjreddie/darknet/a028bfa0da8eb96583c05d5bd00f4bfb9543f2da/data/coco.names")
                dl_stat += "\n  -YOLO LABELS file downloaded."
            except:
                dl_stat += "\n  -An exception occurred while downloading YOLO LABELS file."
        progress_callback.emit(dl_stat)
        if not os.path.isfile(weights_Path):
            dl_stat += "\nYOLO W8s file download started..."
            progress_callback.emit(dl_stat)
            try:
                temporary_downloader(weights_Path, "https://pjreddie.com/media/files/yolov3.weights")
                dl_stat += "\n  -YOLO W8s file downloaded."
            except:
                dl_stat += "\n  -An exception occurred while downloading YOLO W8s file."
        progress_callback.emit(dl_stat)
        if not os.path.isfile(h5_model_path):
            dl_stat += "\nModel download started..."
            progress_callback.emit(dl_stat)
            try:
                temporary_downloader(zip_model_path, "https://s20.picofile.com/d/8447340318/e94262e5-ca04-42f9-84cc-a588c7404960/estimator_model.zip")
                dl_stat += "\n  -Model downloaded. Extraction started..."
                progress_callback.emit(dl_stat)
                with zipfile.ZipFile(zip_model_path) as zf:
                    zf.extractall(res_path)
                    dl_stat += "\n  -Extraction finished."
            except:
                dl_stat += "\n  -An exception occurred while downloading Model file."
        progress_callback.emit(dl_stat)
        dl_stat += "\nAll files are ready!\n\nBrowse the picture and Start the process!"
        progress_callback.emit(dl_stat)
        self.CHECK_BTN.setEnabled(True)
        self.DL_BTN.setEnabled(True)
        self.BRW_BTN.setEnabled(True)
        self.START_BTN.setEnabled(True)

    def download_stat_updater(self, stat):
        self.NTF_LBL.setText(stat)
    
    def dl_function(self):
        worker = Worker(self.downloader)
        worker.signals.progress.connect(self.download_stat_updater)
        self.NTF_LBL.setText(" ")
        self.threadpool.start(worker)

    def get_image(self):
        global file_path
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Single File', QtCore.QDir.rootPath() , "Image files (*.jpg *.png)")
        self.PH_LBL.setPixmap(QtGui.QPixmap(file_path))
        self.NTF_LBL.setText("File path:\n" + file_path)
    
    def do_the_job(self):
        print(file_path)
        # worker = Worker(self.downloader)
        # worker.signals.progress.connect(self.download_stat_updater)
        # self.NTF_LBL.setText(" ")
        # self.threadpool.start(worker)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ObDES"))
        self.CHECK_BTN.setText(_translate("MainWindow", "Check Prerequisites"))
        self.DL_BTN.setText(_translate("MainWindow", "Download Required Files"))
        self.START_BTN.setText(_translate("MainWindow", "Start"))
        self.NTF_LBL.setText(_translate("MainWindow", "Notification Center!"))
        self.PH_LBL.setText(_translate("MainWindow", "Pictures will be shown here."))
        self.BRW_BTN.setText(_translate("MainWindow", "Browse Input File"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
