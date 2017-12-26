#include "tracker.hpp"

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
    object.objectId = personID;

    tracker.addObject(object);
}
void MultiObjectTLDTracker::deleteTarget(int personID)
{
    tracker.deleteObject(personID);
}
void MultiObjectTLDTracker::update(Mat image)
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

void MultiObjectTLDTracker::drawBoxes(Mat image)
{
    std::vector<ObjectBox> boxes = tracker.getObjectBoxes();

    for (int i = 0; i < boxes.size(); i++)
    {
        Rect rec (boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height);
        rectangle(image, rec, Scalar(255, 0, 0), 2, 1 );
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

        recAndID.personID = objectBoxes[i].objectId;

        objects.push_back(recAndID);        
    }

    std::sort(objects.begin(), objects.end());

    return objects;
}