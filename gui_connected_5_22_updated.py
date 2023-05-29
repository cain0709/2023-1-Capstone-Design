import sys
import requests
import openai
import os
import pandas as pd
import numpy as np
import utils
import time
import signal
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QPalette, QColor, QIcon
from datetime import datetime

model_path_list = []

class Worker(QThread):

    output_field_signal = pyqtSignal(str)

    def __init__(self, parent, query_list, csv_path):
        super().__init__(parent)
        self.parent = parent
        self.csv_path = csv_path
        self.query_list = query_list
        self.output_field_text = ''
        self.model_list = ['rf','et', 'xgb', 'lgbm']

    def __del__(self):
        print("API Worker Deleted")

    def run(self):
        model_path_list = []

        print("Worker Run!")
        model_save_path = os.getcwd() + '/generated_model'
        now = datetime.now()     
        timestamp = str(now.timestamp()).split('.')[0]
        csv_name = self.csv_path.split('/')[-1].split('.')[0]

        self.output_field_text += 'Code Generate Start!\n'
        self.output_field_signal.emit(self.output_field_text)

        # Set user_content with the content from query_list
        for i, user_content in enumerate(self.query_list):
            self.output_field_text = self.output_field_text + f"{self.model_list[i]} generating..\n"
            self.output_field_signal.emit(self.output_field_text)
            messages = []
            messages.append({"role": "user", "content": f"{user_content}"})
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

            assistant_content = completion.choices[0].message["content"].strip().replace('`','').replace('python','')
            messages.append({"role": "assistant", "content": f"{assistant_content}"})

            print(f"GPT: {assistant_content}")

            if not os.path.isdir(model_save_path):
                os.mkdir(model_save_path)
            model_path_list.append(f'{model_save_path}/{csv_name}_{self.model_list[i]}_{timestamp}.py')
            file = open(f'{model_save_path}/{csv_name}_{self.model_list[i]}_{timestamp}.py', 'w')    # hello.txt 파일을 쓰기 모드(w)로 열기. 파일 객체 반환
            file.write(assistant_content)      # 파일에 문자열 저장
            file.close()
            self.output_field_text = self.output_field_text + self.model_list[i] + ' model generated finish!\n'
            self.output_field_signal.emit(self.output_field_text)

        self.output_field_text += 'All model created!\n'
        self.output_field_signal.emit(self.output_field_text)

        '''
        print("Model path list : " +str(model_path_list))
        for i, model_py_path in enumerate(model_path_list):
            self.output_field_text = self.output_field_text + model_py_path + ' execute!\n' 
            self.output_field_signal.emit(self.output_field_text)
            cmd = ["python", "-u", model_py_path]
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
            while True:
                output = process.stdout.readline()
                print("OUTPUT : " + str(output.strip()))
                if output == '' and process.poll() is not None or self.isRunning() is False:
                    print("Stop subprocess")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM) 
                    break
                if output:
                    self.output_field_text += output.strip()
                    self.output_field_signal.emit(self.output_field_text)
                    print("TEXT" +str(self.output_field_text))
        '''

        self.__del__()


      

class ChatGPT_GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.prob_type = None #'Classification' or 'Regression'
        self.query_list = None
        self.csv_path = None
        self.target_cols = None
        self.useless_cols = None

        self.util = utils.utils()
        self.initUI()

        # dark mode
        self.dark_mode()

    def initUI(self):
        # Create labels for input fields
        csv_path_label = QLabel('Path of CSV file:')
        model_type_label = QLabel('Model type (Classification/Regression):')
        target_cols_label = QLabel('Target columns (separated by //):')
        useless_cols_label = QLabel('Useless-feature columns (separated by //):')

        # Create input fields
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['Classification', 'Regression'])
        self.target_cols_edit = QTextEdit()
        self.useless_cols_edit = QTextEdit()

        # Adjust the size of target_cols_edit and useless_cols_edit to be the same
        self.target_cols_edit.setMinimumHeight(50)
        self.useless_cols_edit.setMinimumHeight(50)

        # Create select CSV file button
        self.csv_path_btn = QPushButton('Select CSV File', self)
        self.csv_path_btn.clicked.connect(self.select_csv_file)

        # Create send button
        send_btn = QPushButton('Send', self)
        send_btn.clicked.connect(self.send_request)

        # Create output field
        self.output_field = QTextEdit()

        # Create layout for input fields
        input_layout = QVBoxLayout()
        input_layout.addWidget(csv_path_label)
        input_layout.addWidget(self.csv_path_btn)
        input_layout.addWidget(model_type_label)
        input_layout.addWidget(self.model_type_combo)
        input_layout.addWidget(target_cols_label)
        input_layout.addWidget(self.target_cols_edit)
        input_layout.addWidget(useless_cols_label)
        input_layout.addWidget(self.useless_cols_edit)

        # Create layout for send button and output field
        output_layout = QVBoxLayout()
        output_layout.addWidget(send_btn)
        output_layout.addWidget(self.output_field)

        # Create main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)

        # Set main layout
        self.setLayout(main_layout)

        # Set window properties - 1, 2는 생성위치 3, 4는 창 크기
        self.setWindowIcon(QIcon("icon.png"))
        self.setWindowTitle('A.C.G.C')
        self.setGeometry(500, 150, 2000, 1500)

    
    def dark_mode(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(palette)
        self.setPalette(palette)

        # Set the style for QFileDialog, QComboBox, and QPushButton
        style = """
        QFileDialog, QComboBox, QPushButton {
            background-color: #353535;
            color: #ffffff;
            border: 2px solid #76797C;
            padding: 5px;
        }

        QFileDialog:hover, QComboBox:hover, QPushButton:hover {
            border: 2px solid #3daee9;
        }

        QFileDialog:selected, QComboBox:selected, QPushButton:selected {
            background-color: #3daee9;
        }
        """
        self.setStyleSheet(style)
    
    def select_csv_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        if filename:
            with open('input_values.txt', 'w') as f:
                f.write(f"CSV Path: {filename}\n")
            self.csv_path_btn.setText(filename)

    def save_input_values(self, csv_path, model_type, target_cols, useless_cols):
        with open('input_values.txt', 'w') as f:
            f.write(f"CSV Path: {csv_path}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Target Columns: {', '.join(target_cols)}\n")
            f.write(f"Useless-Feature Columns: {', '.join(useless_cols)}\n")
            
            
    def send_request(self):
        # Get input values
        self.output_field_test = ''
        csv_path = self.csv_path_btn.text() 
        model_type = self.model_type_combo.currentText()
        target_cols = self.target_cols_edit.toPlainText()
        useless_cols = self.useless_cols_edit.toPlainText().split('//')


        # Save input values to file
        self.save_input_values(csv_path, model_type, target_cols, useless_cols)

        openai.api_key = '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'

        if model_type == "Classification":
            query_list = self.util.classification_process(csv_path, useless_cols, target_cols)

        elif model_type == "Regression":
            query_list = self.util.regression_process(csv_path, useless_cols, target_cols)

        worker = Worker(self, query_list, csv_path)
        worker.start()
        worker.output_field_signal.connect(self.output_field_print)
        # Update output field with response text


    def output_field_print(self, str):
        self.output_field.setText(str)


    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatgpt_gui = ChatGPT_GUI()
    chatgpt_gui.show()
    sys.exit(app.exec_())
