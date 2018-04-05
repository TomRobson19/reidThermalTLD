import random

class Person:

	def __init__(self, identifier):
		self.personIdentifier = identifier
		random.seed(identifier)
		self.personColour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		self.previous = []
		self.active = True

	def getIdentifier(self):
		return self.personIdentifier

	def getColour(self):
		return self.personColour

	def addPrevious(self, roi):
		if len(self.previous)<5:
			self.previous.append(roi)
		# else:
		# 	self.previous = self.previous[1:]
		# 	self.previous.append(roi)
		self.active = True

	def getPrevious(self):
		return self.previous

	def makeInactive(self):
		self.active = False

	def isActive(self):
		return self.active