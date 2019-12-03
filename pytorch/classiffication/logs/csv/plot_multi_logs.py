import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
 
'''读取csv文件'''
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2])) 
        x.append((row[1]))
    return x ,y
 
 
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
 
 
plt.figure()
x2,y2=readcsv("./csv/lenet_test_acc.csv")
plt.plot(x2, y2, color='red', label='lenet')
#plt.plot(x2, y2, '.', color='red')
 
x,y=readcsv("./csv/mobilenet_test_acc.csv")
plt.plot(x, y, 'g',label='mobilenet')
 
x1,y1=readcsv("./csv/resnet_test_acc.csv")
plt.plot(x1, y1, color='black',label='resnet')
 
x4,y4=readcsv("./csv/densenet_test_acc.csv")
plt.plot(x4, y4, color='blue',label='densenet')
 
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
 
plt.ylim(0, 50)
plt.xlim(0, 30)
plt.xlabel('Steps',fontsize=20)
plt.ylabel('Score',fontsize=20)
plt.legend(fontsize=10)
plt.show()
