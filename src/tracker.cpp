#include "tracker.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <thread>
#include <chrono>
#include <X11/Xlib.h>
#include <unistd.h>

#define imagesFIFO "/tmp/images.fifo" 

MultiObjectTLDTracker::MultiObjectTLDTracker() 
{

}

void MultiObjectTLDTracker::addTarget(Rect boundingBox, int personID)
{
    //convert rectangle to ObjectBox
    ObjectBox object;
    object.x = boundingBox.x;
    object.y = boundingBox.y;
    object.width = boundingBox.width;
    object.height = boundingBox.height;
    object.personID = personID;

    tracker.addObject(object);
}
void MultiObjectTLDTracker::deleteTarget(int personID)
{
    tracker.deleteObject(personID);

    string newLine="\n"; 

    FILE * fifo;

    fifo = fopen (imagesFIFO,"w");

    string person = to_string(personID);

    string toSend = person+newLine;

    fprintf(fifo, "%s", toSend.c_str());

    fclose(fifo);

    // if (tracker.getObjectBoxes().size() == 0)
    // {
    //     cout << "reinit" << endl;
    //     MultiObjectTLD tracker = MultiObjectTLD(1280,960,settings);
    // }
}
void MultiObjectTLDTracker::update(Mat image)
{
    std::vector<ObjectBox> previousObjectBoxes = tracker.getObjectBoxes();

    IplImage frame = (IplImage(image));

    int size = image.cols * image.rows;

    unsigned char img[size];

    for(int j = 0; j < size; j++)
    {
        img[j] = frame.imageData[j];
    }

    tracker.processFrame(img);

    std::vector<ObjectBox> newObjectBoxes = tracker.getObjectBoxes();

    for(int i=0; i<newObjectBoxes.size(); i++)
    {
        for(int j=0; j<previousObjectBoxes.size(); j++)
        {
            if(newObjectBoxes[i].personID == previousObjectBoxes[j].personID)
            {
                if((abs(newObjectBoxes[i].x - previousObjectBoxes[j].x) > 20) || (abs(newObjectBoxes[i].y - previousObjectBoxes[j].y)) > 20)
                {
                    deleteTarget(newObjectBoxes[i].personID);
                }
            }
        }
    }
}

void MultiObjectTLDTracker::drawBoxes(Mat image)
{
    std::vector<ObjectBox> boxes = tracker.getObjectBoxes();

    for (int i = 0; i < boxes.size(); i++)
    {
        Rect rec (boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height);
        int personID = boxes[i].personID;

        srand(personID*10);

        rectangle(image, rec, Scalar(rand() % 255, rand() % 255, rand() % 255), 2, 1 );
    }
}

int MultiObjectTLDTracker::getNumberOfObjects()
{
    return tracker.getObjectBoxes().size();
}

std::vector<rectangleAndID> MultiObjectTLDTracker::getObjectRectangles()
{
    std::vector<ObjectBox> objectBoxes = tracker.getObjectBoxes();
    
    std::vector<rectangleAndID> objects;

    for (int i = 0; i < objectBoxes.size(); i++)
    {
        Rect temp;
        temp.x = objectBoxes[i].x;
        temp.y = objectBoxes[i].y;
        temp.width = objectBoxes[i].width;
        temp.height = objectBoxes[i].height;

        rectangleAndID recAndID;

        recAndID.rectangle = temp;

        recAndID.personID = objectBoxes[i].personID;

        objects.push_back(recAndID);        
    }

    std::sort(objects.begin(), objects.end());

    return objects;
}