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
void MultiObjectTLDTracker::deleteTarget()
{
    //use the getObjectBoxes function
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
    std::vector<ObjectBox> boxes;
    boxes = tracker.getObjectBoxes();

    for (int i = 0; i < boxes.size(); i++)
    {
        Rect rec (boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height);
        rectangle(image, rec, (255, 0, 0), 2, 1 );
    }
}