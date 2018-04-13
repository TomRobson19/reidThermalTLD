import csv
import numpy as np
positiveTrain= []
negativeTrain = []

positiveTest = []
negativeTest = []

positiveEval = []
negativeEval = []

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
         
with open('eval_predictions.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
          row = float(row[0])
          if counter % 2 == 0:
               positiveEval.append(row)
          else:
               negativeEval.append(row)
          counter += 1

np.savetxt("positiveTrain.csv",positiveTrain, delimiter=",")
np.savetxt("negativeTrain.csv",negativeTrain, delimiter=",")

np.savetxt("positiveTest.csv",positiveTest, delimiter=",")
np.savetxt("negativeTest.csv",negativeTest, delimiter=",")

np.savetxt("positiveEval.csv",positiveEval, delimiter=",")
np.savetxt("negativeEval.csv",negativeEval, delimiter=",")