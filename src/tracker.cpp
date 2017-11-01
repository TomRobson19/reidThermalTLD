#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

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
    virtual void initialise(int width, int height);
    virtual void addTarget(Rect boundingBox, int personID);
    virtual void deleteTarget();
    virtual void update(Mat image);
};

class MultiObjectTLDTracker : public GenericTracker
{
public:
	MultiObjectTLDTracker(Mat image) : tracker(image.cols, image.rows, settings) {}
    // void initialise(int width, int height)
    // {
    //     tracker = MultiObjectTLD(width, height, settings);
    // }
    void addTarget(Rect boundingBox, int personID)
    {
    	//convert rectangle to ObjectBox
    	ObjectBox object;
    	object.x = boundingBox.x;
    	object.y = boundingBox.y;
    	object.width = boundingBox.width;
    	object.height = boundingBox.height;
    	object.objectId = personID;

        tracker.addObject(object);
    }
    void deleteTarget()
    {
    	//use the getObjectBoxes function
    }
    void update(Mat image)
    {
        IplImage frame = (IplImage(image));

        int size = image.cols * image.rows;

        unsigned char img[size * 3];

        for(int j = 0; j < size; j++)
        {
            img[j] = frame.imageData[j * 3 + 2];
            img[j + size] = frame.imageData[j * 3 + 1];
            img[j + 2 * size] = frame.imageData[j * 3];
        }
        tracker.processFrame(img);
    }
private:
    MultiObjectTLD tracker;
    MOTLDSettings settings();
};