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

class MultiObjectTLDTracker
{
private:
    const MOTLDSettings settings = MOTLDSettings(0);
    MultiObjectTLD tracker = MultiObjectTLD(1280,960,settings);
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