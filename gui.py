 # -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'emrullah.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
import glob, os
import sys
import subprocess
import shutil
from PyQt5 import QtCore, QtGui, QtWidgets

now_path = os.getcwd()

class Ui_BitirmeProjesi(object):
    def setupUi(self, BitirmeProjesi):
        BitirmeProjesi.setObjectName("BitirmeProjesi")
        BitirmeProjesi.resize(774, 459)
        self.merkezbilesen = QtWidgets.QWidget(BitirmeProjesi)
        self.merkezbilesen.setObjectName("merkezbilesen")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.merkezbilesen)
        self.verticalLayout.setObjectName("verticalLayout")
        self.notification = QtWidgets.QLabel(self.merkezbilesen)
        self.notification.setEnabled(True)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.notification.setFont(font)
        self.notification.setStyleSheet("color: red")
        self.notification.setAlignment(QtCore.Qt.AlignCenter)
        self.notification.setObjectName("notification")
        self.verticalLayout.addWidget(self.notification)
        self.tabWidget = QtWidgets.QTabWidget(self.merkezbilesen)
        self.tabWidget.setObjectName("tabWidget")
        self.test = QtWidgets.QWidget()
        self.test.setEnabled(True)
        self.test.setMaximumSize(QtCore.QSize(744, 16777215))
        self.test.setObjectName("test")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.test)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.modelBaslik = QtWidgets.QLabel(self.test)
        self.modelBaslik.setObjectName("modelBaslik")
        self.gridLayout_8.addWidget(self.modelBaslik, 2, 0, 1, 1)
        self.calistir = QtWidgets.QPushButton(self.test)
        self.calistir.setObjectName("calistir")
        self.gridLayout_8.addWidget(self.calistir, 3, 1, 1, 1)
        self.ikinciCumleBaslik = QtWidgets.QLabel(self.test)
        self.ikinciCumleBaslik.setObjectName("ikinciCumleBaslik")
        self.gridLayout_8.addWidget(self.ikinciCumleBaslik, 1, 0, 1, 1)
        self.ilkCumleDeger = QtWidgets.QLineEdit(self.test)
        self.ilkCumleDeger.setObjectName("ilkCumleDeger")
        self.gridLayout_8.addWidget(self.ilkCumleDeger, 0, 1, 1, 1)
        self.modelDeger = QtWidgets.QComboBox(self.test)
        self.modelDeger.setObjectName("modelDeger")
        self.gridLayout_8.addWidget(self.modelDeger, 2, 1, 1, 1)
        self.ikinciCumleDeger = QtWidgets.QLineEdit(self.test)
        self.ikinciCumleDeger.setObjectName("ikinciCumleDeger")
        self.gridLayout_8.addWidget(self.ikinciCumleDeger, 1, 1, 1, 1)
        self.ilkCumleBaslik = QtWidgets.QLabel(self.test)
        self.ilkCumleBaslik.setObjectName("ilkCumleBaslik")
        self.gridLayout_8.addWidget(self.ilkCumleBaslik, 0, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_8)
        self.ayirac = QtWidgets.QFrame(self.test)
        self.ayirac.setEnabled(True)
        self.ayirac.setFrameShape(QtWidgets.QFrame.HLine)
        self.ayirac.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.ayirac.setObjectName("ayirac")
        self.verticalLayout_2.addWidget(self.ayirac)
        self.sonucBaslik = QtWidgets.QLabel(self.test)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.sonucBaslik.setFont(font)
        self.sonucBaslik.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sonucBaslik.setAutoFillBackground(False)
        self.sonucBaslik.setStyleSheet("")
        self.sonucBaslik.setScaledContents(False)
        self.sonucBaslik.setAlignment(QtCore.Qt.AlignCenter)
        self.sonucBaslik.setWordWrap(False)
        self.sonucBaslik.setObjectName("sonucBaslik")
        self.verticalLayout_2.addWidget(self.sonucBaslik)
        self.sonucDeger = QtWidgets.QLabel(self.test)
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.sonucDeger.setFont(font)
        self.sonucDeger.setAutoFillBackground(False)
        self.sonucDeger.setStyleSheet("color: green;")
        self.sonucDeger.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sonucDeger.setFrameShadow(QtWidgets.QFrame.Plain)
        self.sonucDeger.setScaledContents(False)
        self.sonucDeger.setAlignment(QtCore.Qt.AlignCenter)
        self.sonucDeger.setObjectName("sonucDeger")
        self.verticalLayout_2.addWidget(self.sonucDeger)
        self.tabWidget.addTab(self.test, "")
        self.train = QtWidgets.QWidget()
        self.train.setObjectName("train")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.train)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_2 = QtWidgets.QLabel(self.train)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.train)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 8, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.train)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.dev_file_dropdown = QtWidgets.QComboBox(self.train)
        self.dev_file_dropdown.setObjectName("dev_file_dropdown")
        self.dev_file_dropdown.setItemText(3, "")
        self.gridLayout_3.addWidget(self.dev_file_dropdown, 3, 1, 1, 1)
        self.max_len_input = QtWidgets.QLineEdit(self.train)
        self.max_len_input.setObjectName("max_len_input")
        self.gridLayout_3.addWidget(self.max_len_input, 7, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.train)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 5, 0, 1, 1)
        self.test_file_dropdown = QtWidgets.QComboBox(self.train)
        self.test_file_dropdown.setObjectName("test_file_dropdown")
        self.test_file_dropdown.setItemText(3, "")
        self.gridLayout_3.addWidget(self.test_file_dropdown, 1, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.train)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 6, 0, 1, 1)
        self.batch_size_inpiut = QtWidgets.QLineEdit(self.train)
        self.batch_size_inpiut.setObjectName("batch_size_inpiut")
        self.gridLayout_3.addWidget(self.batch_size_inpiut, 5, 1, 1, 1)
        self.train_file_dropdown = QtWidgets.QComboBox(self.train)
        self.train_file_dropdown.setObjectName("train_file_dropdown")
        self.train_file_dropdown.setItemText(3, "")
        self.gridLayout_3.addWidget(self.train_file_dropdown, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.train)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.epoch_input = QtWidgets.QLineEdit(self.train)
        self.epoch_input.setObjectName("epoch_input")
        self.gridLayout_3.addWidget(self.epoch_input, 8, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.train)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 3, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.train)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 7, 0, 1, 1)
        self.patience_input = QtWidgets.QLineEdit(self.train)
        self.patience_input.setObjectName("patience_input")
        self.gridLayout_3.addWidget(self.patience_input, 6, 1, 1, 1)
        self.lang_dopDown = QtWidgets.QComboBox(self.train)
        self.lang_dopDown.setObjectName("lang_dopDown")
        self.lang_dopDown.addItem("")
        self.lang_dopDown.addItem("")
        self.gridLayout_3.addWidget(self.lang_dopDown, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.save_button = QtWidgets.QPushButton(self.train)
        self.save_button.setObjectName("save_button")
        self.gridLayout_7.addWidget(self.save_button, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.train)
        self.label_4.setObjectName("label_4")
        self.gridLayout_7.addWidget(self.label_4, 0, 0, 1, 1)
        self.model_name_input = QtWidgets.QLineEdit(self.train)
        self.model_name_input.setObjectName("model_name_input")
        self.gridLayout_7.addWidget(self.model_name_input, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_7, 11, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.clear_button = QtWidgets.QPushButton(self.train)
        self.clear_button.setMaximumSize(QtCore.QSize(16777215, 32))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.clear_button.setFont(font)
        self.clear_button.setObjectName("clear_button")
        self.gridLayout_5.addWidget(self.clear_button, 0, 0, 1, 1)
        self.run_button = QtWidgets.QPushButton(self.train)
        self.run_button.setObjectName("run_button")
        self.gridLayout_5.addWidget(self.run_button, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_5, 3, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.acc = QtWidgets.QLabel(self.train)
        self.acc.setMinimumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.acc.setFont(font)
        self.acc.setTextFormat(QtCore.Qt.AutoText)
        self.acc.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.acc.setObjectName("acc")
        self.gridLayout_6.addWidget(self.acc, 0, 0, 1, 1)
        self.text_acc_2 = QtWidgets.QLabel(self.train)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.text_acc_2.setFont(font)
        self.text_acc_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.text_acc_2.setObjectName("text_acc_2")
        self.gridLayout_6.addWidget(self.text_acc_2, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_6, 8, 0, 1, 1)
        self.tabWidget.addTab(self.train, "")
        self.verticalLayout.addWidget(self.tabWidget)
        BitirmeProjesi.setCentralWidget(self.merkezbilesen)
        self.statusbar = QtWidgets.QStatusBar(BitirmeProjesi)
        self.statusbar.setObjectName("statusbar")
        BitirmeProjesi.setStatusBar(self.statusbar)

        self.retranslateUi(BitirmeProjesi)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(BitirmeProjesi)

    def retranslateUi(self, BitirmeProjesi):
        _translate = QtCore.QCoreApplication.translate
        BitirmeProjesi.setWindowTitle(_translate("BitirmeProjesi", "Bitirme Projesi"))
        self.notification.setText(_translate("BitirmeProjesi", "TextLabel"))
        self.modelBaslik.setText(_translate("BitirmeProjesi", "Choose Model"))
        self.calistir.setText(_translate("BitirmeProjesi", "Run Test"))
        self.ikinciCumleBaslik.setText(_translate("BitirmeProjesi", "Second Sentence"))
        self.ilkCumleBaslik.setText(_translate("BitirmeProjesi", "First Sentence"))
        self.sonucBaslik.setText(_translate("BitirmeProjesi", "Results"))
        self.sonucDeger.setText(_translate("BitirmeProjesi", "-"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.test), _translate("BitirmeProjesi", "Test"))
        self.label_2.setText(_translate("BitirmeProjesi", "Train File"))
        self.label_13.setText(_translate("BitirmeProjesi", "Epoch"))
        self.label_5.setText(_translate("BitirmeProjesi", "Language"))
        self.dev_file_dropdown.setItemText(0, _translate("BitirmeProjesi", "en_test"))
        self.dev_file_dropdown.setItemText(1, _translate("BitirmeProjesi", "tr_test"))
        self.dev_file_dropdown.setItemText(2, _translate("BitirmeProjesi", "our_test"))
        self.label_10.setText(_translate("BitirmeProjesi", "Batch_size"))
        self.test_file_dropdown.setItemText(0, _translate("BitirmeProjesi", "en_test"))
        self.test_file_dropdown.setItemText(1, _translate("BitirmeProjesi", "tr_test"))
        self.test_file_dropdown.setItemText(2, _translate("BitirmeProjesi", "our_test"))
        self.label_11.setText(_translate("BitirmeProjesi", "Patience"))
        self.train_file_dropdown.setItemText(0, _translate("BitirmeProjesi", "en_test"))
        self.train_file_dropdown.setItemText(1, _translate("BitirmeProjesi", "tr_test"))
        self.train_file_dropdown.setItemText(2, _translate("BitirmeProjesi", "our_test"))
        self.label.setText(_translate("BitirmeProjesi", "Test File"))
        self.label_3.setText(_translate("BitirmeProjesi", "Dev File"))
        self.label_12.setText(_translate("BitirmeProjesi", "Max_length"))
        self.lang_dopDown.setItemText(0, _translate("BitirmeProjesi", "tr"))
        self.lang_dopDown.setItemText(1, _translate("BitirmeProjesi", "en"))
        self.save_button.setText(_translate("BitirmeProjesi", "Save"))
        self.label_4.setText(_translate("BitirmeProjesi", "Model Name"))
        self.clear_button.setText(_translate("BitirmeProjesi", "Default Values"))
        self.run_button.setText(_translate("BitirmeProjesi", "Run"))
        self.acc.setText(_translate("BitirmeProjesi", "Loss / Accuracy(%):"))
        self.text_acc_2.setText(_translate("BitirmeProjesi", "-"))
        self.notification.hide()
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.train), _translate("BitirmeProjesi", "Train"))

        ###

        os.chdir(now_path + "/models")
        i = 0
        for file in glob.glob("*.h5"):
            self.modelDeger.addItem("")
            self.modelDeger.setItemText(i, _translate("BitirmeProjesi", file))
            i += 1
        ###
        self.calistir.clicked.connect(self.run)
        self.run_button.clicked.connect(self.trainFunc)
        self.lang_dopDown.currentIndexChanged.connect(self.onLangChance)
        self.save_button.clicked.connect(self.saveModels)
        self.clear_button.clicked.connect(self.defaultValues)
        
        
        
        self.dosyalariCek('tr')

        self.defaultValues()

        
        #######3
    def saveModels(self):
        os.chdir(now_path)
        if ( os.path.isfile('./newModel.h5') ):
            yeniIsim = self.model_name_input.text()
            os.rename("./newModel.h5", yeniIsim + ".h5")
            os.rename("./newModel.npy", yeniIsim + ".npy")
            shutil.move(yeniIsim + ".h5", "models/" + yeniIsim + ".h5")
            shutil.move(yeniIsim + ".npy", "models/" + yeniIsim + ".npy")
            self.modelDeger.clear()
            os.chdir(now_path + "/models")
            i = 0
            for file in glob.glob("*.h5"):
                self.modelDeger.addItem("")
                self.modelDeger.setItemText(i, file)
                i += 1

    def defaultValues(self):
        self.batch_size_inpiut.setText('512')
        self.patience_input.setText('4')
        self.max_len_input.setText('42')
        self.epoch_input.setText('1')

    def dosyalariCek(self, lang_param):
        self.test_file_dropdown.clear()
        self.train_file_dropdown.clear()
        self.dev_file_dropdown.clear()
        if lang_param == 'tr':
            ######## drop down files
            os.chdir(now_path + "/testFiles")
            i = 0
            for file in glob.glob("*.csv"):
                self.test_file_dropdown.addItem("")
                self.test_file_dropdown.setItemText(i,file)
                i += 1
            os.chdir(now_path + "/trainFiles")
            i = 0
            for file in glob.glob("*.csv"):
                self.train_file_dropdown.addItem("")
                self.train_file_dropdown.setItemText(i, file)
                i += 1
            os.chdir(now_path + "/devFiles")
            i = 0
            for file in glob.glob("*.csv"):
                self.dev_file_dropdown.addItem("")
                self.dev_file_dropdown.setItemText(i,file)
                i += 1
        else:
            ######## drop down files
            os.chdir(now_path + "/testFiles")
            i = 0
            for file in glob.glob("*.jsonl"):
                self.test_file_dropdown.addItem("")
                self.test_file_dropdown.setItemText(i,  file)
                i += 1
            os.chdir(now_path + "/trainFiles")
            i = 0
            for file in glob.glob("*.jsonl"):
                self.train_file_dropdown.addItem("")
                self.train_file_dropdown.setItemText(i,  file)
                i += 1
            os.chdir(now_path + "/devFiles")
            i = 0
            for file in glob.glob("*.jsonl"):
                self.dev_file_dropdown.addItem("")
                self.dev_file_dropdown.setItemText(i, file)
                i += 1
    def onLangChance(self):
        self.dosyalariCek(self.lang_dopDown.currentText())
    def trainFunc(self):
        self.notification.hide()
        lang_value = self.lang_dopDown.currentText()
        test_file_value = self.test_file_dropdown.currentText()
        train_file_value = self.train_file_dropdown.currentText()
        dev_file_value = self.dev_file_dropdown.currentText()
        batch_size_value = self.batch_size_inpiut.text()
        patience_value = self.patience_input.text()
        max_len_value = self.max_len_input.text()
        epoch_value = self.epoch_input.text()

        if batch_size_value.strip() != "" and patience_value.strip() != "" and max_len_value.strip() != "" and epoch_value.strip() != "":
            if lang_value == 'tr':
                os.chdir(now_path)
                subprocess.call("python3 snli_translated.py " + str(test_file_value) + " " + str(train_file_value) + " " + str(dev_file_value) + " " + batch_size_value + " " + patience_value + " " + max_len_value + " " + epoch_value, shell=True)
            
            else:
                os.chdir(now_path)
                subprocess.call("python3 snli_rnn.py " + str(test_file_value) + " " + str(train_file_value) + " " + str(dev_file_value) + " " + batch_size_value + " " + patience_value + " " + max_len_value + " " + epoch_value, shell=True)
            
            with open('train_results.txt') as fp:
                line = fp.readline()
                self.text_acc_2.setText(line)
        else:
            self.notification.setText("Bos parametre birakmayiniz...")
            self.notification.show()

        

    def run(self):
        self.notification.hide()
        self.notification.setText('-')
        cumle1 = self.ilkCumleDeger.text()
        cumle2 = self.ikinciCumleDeger.text()
        modelIsim = self.modelDeger.currentText()
        
        cumle1 = cumle1.strip()
        cumle2 = cumle2.strip()
        cumle1 = cumle1.replace(' ', '_')
        cumle2 = cumle2.replace(' ', '_')

        self.sonucDeger.setText('-')

        os.chdir(now_path)
        subprocess.call("python3 modelOku.py " + cumle1 + " " + cumle2 + " " + str(modelIsim), shell=True)

        with open('results.txt') as fp:
            line = fp.readline()
            if line == '0':
                self.notification.show()
                self.notification.setText('En az bir cumle girin...')
            else:
                dosyaOranlar = fp.readline()
                dosyaSonuc = fp.readline()
                self.sonucDeger.setText(dosyaSonuc + '\n' + dosyaOranlar)
    
    

    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    BitirmeProjesi = QtWidgets.QMainWindow()
    ui = Ui_BitirmeProjesi()
    ui.setupUi(BitirmeProjesi)
    BitirmeProjesi.show()
    sys.exit(app.exec_())

