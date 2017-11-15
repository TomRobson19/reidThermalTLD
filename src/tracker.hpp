#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>

#include "MultiObjectTLD/motld/MultiObjectTLD.hpp"

using namespace cv;
using namespace std;

class GenericTracker
{
public:
	virtual ~GenericTracker();
    virtual void initialise(int width, int height);
    virtual void addTarget(Rect boundingBox, int personID);
    virtual void deleteTarget(int personID);
    virtual void update(Mat image);
    virtual void drawBoxes(Mat image);
    virtual int getNumberOfObjects();
    virtual std::tuple<std::vector<Rect>,std::vector<int>> getObjectRectangles();

};

class MultiObjectTLDTracker// : public GenericTracker
{
private:
    const MOTLDSettings settings = MOTLDSettings(0);
    MultiObjectTLD tracker = MultiObjectTLD(640,480,settings);
public:
    MultiObjectTLDTracker(); 
    void initialise(int width, int height);
    void addTarget(Rect boundingBox, int personID);
    void deleteTarget(int personID);
    void update(Mat image);
    void drawBoxes(Mat image);
    int getNumberOfObjects();
    std::tuple<std::vector<Rect>,std::vector<int>> getObjectRectangles();
};

#endif