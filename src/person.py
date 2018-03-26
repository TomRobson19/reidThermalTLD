import random

class Person:
	personIdentifier = 0
	personColour = (0,0,0)
	previous = []
	active = True

	def __init__(self, identifier):
		self.setIdentifier(identifier)
		self.setColour(identifier)

	def setIdentifier(self, identifier):
		self.personIdentifier = identifier

	def getIdentifier(self):
		return self.personIdentifier

	def setColour(self, identifier):
		random.seed(identifier)
		personColour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

	def getColour(self):
		return self.personColour

	def addPrevious(self, roi):
		if len(self.previous)<10:
			self.previous.append(roi)
		else:
			self.previous = self.previous[1:]
			self.previous.append(roi)
		self.active = True

	def getPrevious(self):
		return self.previous

	def makeInactive(self):
		self.active = False

	def isActive(self):
		return self.active