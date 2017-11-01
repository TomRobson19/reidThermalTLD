#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace ml;

#include "person.hpp"

Person::Person(int identifier, float x, float y, int timeSteps, float w, float h) {
	setIdentifier(identifier);
  setColour(identifier);
	initKalman(x,y,timeSteps,w,h);
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

void Person::initKalman(float x, float y, int timeSteps, float w, float h) {
  KF.init(6, 6, 0);  //position(x,y) velocity(x,y) rectangle(h,w)

  measurement(0) = x;
  measurement(1) = y;
  measurement(2) = 0.0;
  measurement(3) = 0.0;
  measurement(4) = w;
  measurement(5) = h;

  KF.statePre.at<float>(0, 0) = x;
  KF.statePre.at<float>(1, 0) = y;
  KF.statePre.at<float>(2, 0) = 0.0;
  KF.statePre.at<float>(3, 0) = 0.0;
  KF.statePre.at<float>(4, 0) = w;
  KF.statePre.at<float>(5, 0) = h;

  KF.statePost.at<float>(0, 0) = x;
  KF.statePost.at<float>(1, 0) = y; 
  KF.statePost.at<float>(2, 0) = 0.0;
  KF.statePost.at<float>(3, 0) = 0.0;
  KF.statePost.at<float>(4, 0) = w;
  KF.statePost.at<float>(5, 0) = h;

  KF.transitionMatrix = (Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,
                                              0, 1, 0, 1, 0, 0,
                                              0, 0, 1, 0, 0, 0,
                                              0, 0, 0, 1, 0, 0,
                                              0, 0, 0, 0, 1, 0,
                                              0, 0, 0, 0, 0, 1);
  setIdentity(KF.measurementMatrix);

  setIdentity(KF.processNoiseCov, Scalar::all(0.03)); //adjust this for faster convergence - but higher noise
  //small floating point errors present

  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(0.1));

  lastSeen = timeSteps;
}

void Person::kalmanCorrect(float x, float y, int timeSteps, float w, float h) {
  float currentX = measurement(0);
  float currentY = measurement(1);

  int timeGap = timeSteps-lastSeen;

  if(timeGap == 0) //come up with a better way to do this, for now deals with multiple detections in the same timestep
  {
    timeGap = 1;
  }

  measurement(0) = x;
  measurement(1) = y;
  measurement(2) = (float) ((x - currentX)/timeGap);
  measurement(3) = (float) ((y - currentY)/timeGap);
  measurement(4) = w;
  measurement(5) = h;

  Mat estimated = KF.correct(measurement);

  lastSeen = timeSteps;

  history.push_back(measurement);
}

Rect Person::kalmanPredict() {
  Mat prediction = KF.predict();

  Point2f topLeft(prediction.at<float>(0)-prediction.at<float>(4)/2,prediction.at<float>(1)+prediction.at<float>(5)/2);

  Point2f bottomRight(prediction.at<float>(0)+prediction.at<float>(4)/2,prediction.at<float>(1)-prediction.at<float>(5)/2);

  Rect kalmanRect(topLeft, bottomRight);

  KF.statePre.copyTo(KF.statePost);
  KF.errorCovPre.copyTo(KF.errorCovPost);

  return kalmanRect;
}

void Person::updateFeatures(Mat newFeature) 
{
  if(allFeatures.rows == 0)
  {
    allFeatures.push_back(newFeature); 
  }

  else if(norm(newFeature,allFeatures.row(allFeatures.rows-1),NORM_L1) != 0)
  {
    if(allFeatures.rows < 10)
    {
      allFeatures.push_back(newFeature); 
    }
    else
    {
      allFeatures.push_back(newFeature); 
      allFeatures(Range(1, allFeatures.rows), Range(0, allFeatures.cols)).copyTo(allFeatures);
    }
  }
}

Mat Person::getFeatures(){
  return allFeatures;
}
