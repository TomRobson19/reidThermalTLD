#include "person.hpp"

Person::Person(int identifier) {
	setIdentifier(identifier);
 	setColour(identifier);

 	//initialise MOTLD tracker
}

void Person::setColour(int identifier) {
	srand(identifier*100);
 	personColour = Scalar(rand() % 256, rand() % 256, rand() % 256);
}

Scalar Person::getColour() {
	return personColour;
}

void Person::setIdentifier(int identifier) {
	personIdentifier = identifier;
}

int Person::getIdentifier() {
 	return personIdentifier;
}

int Person::getLastSeen() {
	return lastSeen;
}

void Person::setCurrentCamera(int cameraID) {
 	currentCamera = cameraID;
}

int Person::getCurrentCamera() {
 	return currentCamera;
}

// void Person::addTLDObject(Rect boundingBox, MultiObjectTLDTracker tracker) {
// 	tracker.addTarget(boundingBox, personIdentifier);
// }

// void Person::deleteTLDObject(MultiObjectTLDTracker tracker) {
// 	tracker.deleteTarget(personIdentifier);
// }



//implement these in NN phase

// void Person::savePositivePatch(MultiObjectTLDTracker tracker) {

// }

// Rect Person::getPositivePatches(MultiObjectTLDTracker tracker) {
	
// }
//maybe compare new result with 10? previous patches using the CNN and go with majority vote?? 