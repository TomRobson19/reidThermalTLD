import csv
import numpy as np
positiveArray = []
negativeArray = []

with open('te_predictions_aug2.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
     	row = float(row[0])
     	if counter % 2 == 0:
     		positiveArray.append(row)
     	else:
     		negativeArray.append(row)
     	counter += 1

with open('tr_predictions_aug2.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     counter = 0
     for row in reader:
     	row = float(row[0])
     	if counter % 2 == 0:
     		positiveArray.append(row)
     	else:
     		negativeArray.append(row)
     	counter += 1
         

np.savetxt("positive_aug2.csv",positiveArray, delimiter=",")
np.savetxt("negative_aug2.csv",negativeArray, delimiter=",")