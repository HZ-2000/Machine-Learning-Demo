#------------------------------------------#
# Creator: Zach Heimbigner                 #
# Last Modified: 7/23/2021                 #
# Language: Python 3                       #
# Command: python3 App_1.1.py              #
# Notes:                                   #
#  - data tab completed with weight output #
#  - current pattern added to data tab     #
#------------------------------------------#

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from numpy import array, random, zeros
from NeuralNet import NeuralNetwork

class Window(QWidget):
   def __init__(self):
      super(Window, self).__init__()
      self.num_neurons = 2
      self.pattern = array([1,0])
      self.diagram = NN_Diagram()
      self.data = array([])
      self.targets = array([])
      self.NN = NeuralNetwork(self.num_neurons)
      self.answer = 0
      self.initUI()

   def initUI(self):
      hbox = QHBoxLayout(self)
      tabs = QTabWidget()
      tabs.addTab(self.controlTabUI(), "Controls")
      tabs.addTab(self.dataTabUI(), "Data")

      diagram = QFrame()
      box = QVBoxLayout()
      diagram.setLayout(box)
      box.addWidget(self.diagram)

      splitter1 = QSplitter(Qt.Horizontal)
      splitter1.addWidget(tabs)
      splitter1.addWidget(diagram)
      splitter1.setSizes([1000,1000])

      hbox.addWidget(splitter1)

      self.setLayout(hbox)
      QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

      self.setGeometry(300, 300, 1000, 500)
      self.setWindowTitle('NN Demo')
      self.show()

   def controlTabUI(self):
      """Create the Controls page UI."""
      controlTab = QWidget()

      # number of neurons buttons
      layout1 = QHBoxLayout()
      b1 = QPushButton("2")
      b2 = QPushButton("3")
      b3 = QPushButton("4")
      b1.clicked.connect(lambda:self.neuron_change(2))
      b2.clicked.connect(lambda:self.neuron_change(3))
      b3.clicked.connect(lambda:self.neuron_change(4))
      layout1.addWidget(QLabel("# of Neurons:"))
      layout1.addWidget(b1)
      layout1.addWidget(b2)
      layout1.addWidget(b3)

      #new pattern/new dataset buttons
      layout2 = QHBoxLayout()
      b4 = QPushButton("New Pattern")
      b4.clicked.connect(lambda:self.new_pattern())
      b5 = QPushButton("New Data Set")
      b5.clicked.connect(lambda:self.new_data())
      layout2.addWidget(b4)
      layout2.addWidget(b5)

      #display pattern/data/targets
      layout3 = QVBoxLayout()
      self.l1 = QLabel("Current Pattern: " + str(self.pattern))
      self.l2 = QLabel("Data:")
      self.l3 = QLabel("Targets:")
      layout3.addWidget(self.l1)
      layout3.addWidget(self.l2)
      layout3.addWidget(self.l3)

      #complete tab layout
      layout = QVBoxLayout()
      layout.addLayout(layout1)
      layout.addLayout(layout2)
      layout.addLayout(layout3)
      controlTab.setLayout(layout)
      return controlTab

   def dataTabUI(self):
      """Create the Data page UI."""
      dataTab = QWidget()

      #process button
      layout1 = QHBoxLayout()
      b1 = QPushButton("Process")
      b1.clicked.connect(lambda:self.process())
      layout1.addWidget(b1)

      #data display
      layout2 = QVBoxLayout()
      self.l4 = QLabel("Answer: " + str(self.answer))
      self.l5 = QLabel("weights:")
      self.l6 = QLabel("Current Pattern: " + str(self.pattern))
      layout2.addWidget(self.l6)
      layout2.addWidget(self.l4)
      layout2.addWidget(self.l5)

      #create tab
      layout = QVBoxLayout()
      layout.addLayout(layout1)
      layout.addLayout(layout2)
      dataTab.setLayout(layout)
      return dataTab

   def neuron_change(self, x):
      self.num_neurons = x
      self.NN = NeuralNetwork(x)
      self.new_pattern()
      self.new_data()
      self.diagram.num_neurons = x
      self.diagram.repaint()

   def new_pattern(self):
      self.pattern = random.binomial(1, 0.5, self.num_neurons)
      self.l1.setText("Current Pattern: " + str(self.pattern))
      self.l6.setText("Current Pattern: " + str(self.pattern))

   def getData(self):
         A = random.binomial(1, 0.5, (6 ,self.num_neurons))
         B = zeros(6, dtype = int)
         for line, i in zip(A, range(6)):
             B[i] = line[0]
         return A, B

   def new_data(self):
      self.data, self.targets = self.getData()
      self.l2.setText("Data set: \n" + str(self.data))
      self.l3.setText("Targets: \n" + str(self.targets))
      self.NN.train(self.data, self.targets)

   def process(self):
       self.answer = self.NN.feed_forward(self.pattern)
       self.l4.setText("Answer: \n" + str(self.answer))

       weights = ""
       for neuron, i in zip(self.NN.hidden, range(self.num_neurons)):
           weights += "H" + str(i+1) + ": " + str(neuron.weights) + "\n"
       weights += "O1: " + str(self.NN.output.weights)
       self.l5.setText("Weights: \n" + weights)

class NN_Diagram(QWidget):
    def __init__(self):
        super().__init__()
        self.num_neurons = 2

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.paintDiagram(qp)
        qp.end()

    def paintDiagram(self, qp):
        qp.setBrush(QColor(200, 162, 200))  # lilac
        qp.setPen(QColor(200, 162, 200))
        geo = self.geometry()
        height = self.height()
        width = self.width()

        circDiameter = round(height / ((self.num_neurons * 2) + 1))
        circCenter = round(circDiameter / 2)

        #NN Diagram
        #output node
        qp.drawEllipse(width - circDiameter, round(height / 2) - circCenter, circDiameter, circDiameter)

        for i in range(1,((self.num_neurons * 2) + 1),2):
            #hidden layer
            qp.drawEllipse(round(width / 2) - circCenter, (circDiameter * i), circDiameter, circDiameter)
            #input layer
            qp.drawEllipse(0, (circDiameter * i), circDiameter, circDiameter)

        #draw lines for NN
        for i in range(1,((self.num_neurons * 2) + 1),2):
            for k in range(1,((self.num_neurons * 2) + 1),2):
                qp.drawLine(0 + circDiameter, (circDiameter * i) + circCenter, round(width / 2) - circCenter, (circDiameter * k) + circCenter)
            qp.drawLine(round(width / 2) + circCenter, (circDiameter * i) + circCenter, width - circDiameter, round(height / 2))

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = Window()
   sys.exit(app.exec_())
