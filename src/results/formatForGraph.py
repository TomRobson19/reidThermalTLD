import csv
import numpy as np
positiveTrain= []
negativeTrain = []

positiveTest = []
negativeTest = []

positiveVal = []
negativeVal = []

with open('te_predictions.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
     	row = float(row[0])
     	if counter % 2 == 0:
     		positiveTest.append(row)
     	else:
     		negativeTest.append(row)
     	counter += 1

with open('tr_predictions.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
     	row = float(row[0])
     	if counter % 2 == 0:
     		positiveTrain.append(row)
     	else:
     		negativeTrain.append(row)
     	counter += 1
         
with open('val_predictions.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
          row = float(row[0])
          if counter % 2 == 0:
               positiveVal.append(row)
          else:
               negativeVal.append(row)
          counter += 1

np.savetxt("positiveTrain.csv",positiveTrain, delimiter=",")
np.savetxt("negativeTrain.csv",negativeTrain, delimiter=",")

np.savetxt("positiveTest.csv",positiveTest, delimiter=",")
np.savetxt("negativeTest.csv",negativeTest, delimiter=",")

np.savetxt("positiveVal.csv",positiveVal, delimiter=",")
np.savetxt("negativeVal.csv",negativeVal, delimiter=",")