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
