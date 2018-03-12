import random

class Person:
	personIdentifier = 0
	personColour = (0,0,0)
	previous = []

	def __init__(self, identifier):
		setIdentifier(identifier)
		setColour(identifier)

	def setIdentifier(self, identifier):
		personIdentifier = identifier

	def getIdentifier(self):
		return personIdentifier

	def setColour(self, identifier):
		personColour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

	def getColour(self):
		return personColour

	def addPrevious(self, roi):
		if len(previous<10):
			previous.append(roi)
		else:
			previous = previous[1:]
			previous.append(roi)

	def getPrevious(self):
		return previous


#when calling keras, think I just need to convert img to numpy using np.array, astype('float32') and /=255
#should be in this format when calling addPrevious