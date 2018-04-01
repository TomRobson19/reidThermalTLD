#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <stdlib.h>

#include "MultiObjectTLD/motld/MultiObjectTLD.hpp"

using namespace cv;
using namespace std;

struct rectangleAndID {
    Rect rectangle;
    int personID;

    bool operator< (const rectangleAndID &other) const {
        return personID < other.personID;
    }
};

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
    MultiObjectTLD tracker = MultiObjectTLD(1280,960,settings);
    int deletionCounter = 0;
    int currentHighest = -1;
public:
    MultiObjectTLDTracker(); 
    void initialise(int width, int height);
    void addTarget(Rect boundingBox, int personID);
    void deleteTarget(int personID);
    void update(Mat image);
    void drawBoxes(Mat image);
    int getNumberOfObjects();
    std::vector<rectangleAndID> getObjectRectangles();
    int finalDeletionCounter();
};

#endif