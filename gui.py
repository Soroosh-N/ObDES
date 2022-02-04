from PyQt5 import QtCore, QtGui, QtWidgets


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

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.CHECK_BTN.clicked.connect(self.check_function)
        self.DL_BTN.clicked.connect(self.dl_function)

    def check_function(self):
        self.PH_LBL.setPixmap(QtGui.QPixmap("test.png"))

    def dl_function(self):
        self.PH_LBL.setPixmap(QtGui.QPixmap("test.png"))

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
