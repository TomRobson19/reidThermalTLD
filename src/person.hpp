#ifndef PERSON_H
#define PERSON_H

#include <opencv2/opencv.hpp>

#include "tracker.hpp"

using namespace cv;
using namespace std;

class Person
{
private: 
	int personIdentifier;
	Scalar personColour;
	int currentCamera;
	vector<cv::Mat_<float> > history;
	int lastSeen;
	cv::Mat_<float> measurement = cv::Mat_<float>(6,1);



public:

	bool operator< (const Person &other) const {
        return personIdentifier < other.personIdentifier;
    }
	Person(int identifier);

	void setIdentifier(int identifier);

	int getIdentifier();

	void setColour(int identifier);

	Scalar getColour();

	int getLastSeen();

	void setCurrentCamera(int cameraID);

	int getCurrentCamera();

	void addTLDObject(Rect boundingBox, MultiObjectTLDTracker tracker);

	void deleteTLDObject(MultiObjectTLDTracker tracker);

	void savePositivePatch(MultiObjectTLDTracker tracker);

	Rect getPositivePatches(MultiObjectTLDTracker tracker);
	//maybe compare new result with 10? previous patches using the CNN and go with majority vote?? 
};
 
#endif