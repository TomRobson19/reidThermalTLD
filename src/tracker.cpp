#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <algorithm>

#include <stdlib.h>
#include <stdio.h>

#include "cv.h"
#include "highgui.h"

#include "motld/MultiObjectTLD.h"

using namespace cv;
using namespace std;

class GenericTracker
{
public:
    virtual void initialise();
    virtual void addTarget();
    virtual void deleteTarget();
    virtual void update();
};

class MultiObjectTLDTracker : public GenericTracker
{
public:
    void initialise(int width, int height)
    {
        tracker(width, height, settings);
    }
    void addTarget(Rectangle boundingBox)
    {
    	//convert rectangle to ObjectBox
        tracker.addObject(boundingBox);
    }
    void deleteTarget()
    {
    	//use the getObjectBoxes function
    }
    void update(Mat image)
    {
        IplImage *frame = &(IplImage(image));

        unsigned char img[size * 3];

        for(int j = 0; j < size; j++)
        {
            img[j] = frame->imageData[j * 3 + 2];
            img[j + size] = frame->imageData[j * 3 + 1];
            img[j + 2 * size] = frame->imageData[j * 3];
        }
        tracker.processFrame();
    }
private:
    MultiObjectTLD tracker;
    MOTLDSettings settings();
};