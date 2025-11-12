from PyQt5.QtWidgets import  QDialog, QVBoxLayout,QGridLayout
from PyQt5.QtWidgets import  QLabel,QTextBrowser
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

#提示当前进程正在运行的对话框类,用于提示较为复杂的进程
#实时显示控制台信息
class EmittingStr(QtCore.QObject):  
        textWritten = QtCore.pyqtSignal(str)  #定义一个发送str的信号
        def write(self, text):
            self.textWritten.emit(str(text))
        #及时清理缓存避免主窗口关闭时报错
        def flush(self):
            pass



#等待窗口
#显示当前进程的控制台信息
class prompot_Dialog(QDialog):
    def __init__(self):
        super().__init__()
        self.v_layout = QVBoxLayout()#垂直布局
        self.v_layout.setSpacing(0)
        self.setWindowTitle('当前进程提示')
        self.resize(800, 800)
        self.label = QLabel(self)
        self.text_browser =QTextBrowser(self)
        self.layout_init()
        self.font_init()
        #重定向输出
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
   
    def layout_init(self):
        self.v_layout.addWidget(self.label)
        self.v_layout.addWidget(self.text_browser)
        self.setLayout(self.v_layout)

    def font_init(self):
        font_1 = QtGui.QFont('SimSun', 18)
        self.label.setFont(font_1)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        font_2 = QtGui.QFont('SimSun', 14)
        self.text_browser.setFont(font_2)
        
    def setlabel(self,prompt_str):
        self.label.setText(prompt_str)




    def outputWritten(self, text):
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text_browser.setTextCursor(cursor)
        self.text_browser.ensureCursorVisible()

#提示是否选取当前为控制点
#点击确认按钮传出是信号
#点击否按钮传出否信号



class contrlpoint_Dialog(QtWidgets.QDialog):
    signal=QtCore.pyqtSignal(bool) 
    def __init__(self):
        super().__init__()
        self.v_layout = QGridLayout()#垂直布局
        self.v_layout.setSpacing(10)
        self.setWindowTitle('提示信息')
        self.resize(100, 100)
        self.label = QLabel(self)
        self.label.setText('请确认是否选取当前点为控制点')
        self.button_1=QtWidgets.QPushButton(self)
        self.button_1.setText('确认')
        self.button_1.clicked.connect(self.button1_click)
        self.button_2=QtWidgets.QPushButton(self)
        self.button_2.setText('取消')
        self.button_2.clicked.connect(self.button2_click)
        self.layout_init()

   
    def layout_init(self):

        self.v_layout.addWidget(self.label,1,1,1,4)
        self.v_layout.addWidget(self.button_1,2,2,2,1)
        self.v_layout.addWidget(self.button_2,2,3,2,1)
        # self.v_layout.addWidget(self.movie_label)
        self.setLayout(self.v_layout)

    def show_dia(self):
        self.exec_()

    def button1_click(self):
        self.signal.emit(True)
        self.close()

    def button2_click(self):
        self.signal.emit(False)
        self.close()




if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    win=contrlpoint_Dialog()
    win.show_dia()
    




    
    


        














