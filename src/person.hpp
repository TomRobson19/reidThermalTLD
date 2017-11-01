#ifndef PERSON_H
#define PERSON_H

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
 
class Person
{
private: 
	int personIdentifier;
	Scalar personColour;
	int currentCamera;
	vector<cv::Mat_<float> > history;
	int lastSeen;
	cv::KalmanFilter KF;
	cv::Mat_<float> measurement = cv::Mat_<float>(6,1);

	Mat allFeatures;

public:
	Person(int identifier, float x, float y, int timeSteps, float w, float h);

	void setIdentifier(int identifier);

	int getIdentifier();

	void setColour(int identifier);

	Scalar getColour();

	int getLastSeen();

	Point2f getLastPosition();

	void setCurrentCamera(int cameraID);

	int getCurrentCamera();

	void initKalman(float x, float y, int timeSteps, float w, float h);

	void kalmanCorrect(float x, float y, int timeSteps, float w, float h);

	Rect kalmanPredict();

	void updateFeatures(Mat newFeature);

	Mat getFeatures();
};
 
#endif