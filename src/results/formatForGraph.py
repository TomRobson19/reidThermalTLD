import csv
import numpy as np
positiveArray = []
negativeArray = []

positiveEval = []
negativeEval = []

with open('te_predictions.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
     	row = float(row[0])
     	if counter % 2 == 0:
     		positiveArray.append(row)
     	else:
     		negativeArray.append(row)
     	counter += 1

with open('tr_predictions.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
     	row = float(row[0])
     	if counter % 2 == 0:
     		positiveArray.append(row)
     	else:
     		negativeArray.append(row)
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

np.savetxt("positive.csv",positiveArray, delimiter=",")
np.savetxt("negative.csv",negativeArray, delimiter=",")

np.savetxt("positiveEval.csv",positiveEval, delimiter=",")
np.savetxt("negativeEval.csv",negativeEval, delimiter=",")