#include "person.hpp"

Person::Person(int identifier, float x, float y, int timeSteps, float w, float h) {
	setIdentifier(identifier);
  setColour(identifier);
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

Point2f Person::getLastPosition() {
	float currentX = measurement(0);
	float currentY = measurement(1);
	Point2f position = Point2f(currentX,currentY);
	return position;
}

void Person::setCurrentCamera(int cameraID)
{
  currentCamera = cameraID;
}

int Person::getCurrentCamera()
{
  return currentCamera;
}
