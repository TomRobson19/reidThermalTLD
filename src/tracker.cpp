#include "tracker.hpp"
#include <tuple>

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
    std::vector<ObjectBox> objects = tracker.getObjectBoxes();
    for (int i = 0; i < objects.size(); i++)
    {
        if(objects[i].objectId == personID)
        {
            objects.erase(objects.begin()+i);
            i--;
        }
    }
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
        rectangle(image, rec, (255, 0, 0), 2, 1 );
    }
}

int MultiObjectTLDTracker::getNumberOfObjects()
{
    return tracker.getObjectBoxes().size();
}

std::tuple<std::vector<Rect>,std::vector<int>> MultiObjectTLDTracker::getObjectRects()
{
    std::vector<ObjectBox> objectBoxes = tracker.getObjectBoxes();
    std::vector<Rect> rectangles;
    std::vector<int> personIDs;

    for (int i = 0; i < objectBoxes.size(); i++)
    {
        Rect temp;
        temp.x = objectBoxes[i].x;
        temp.y = objectBoxes[i].y;
        temp.width = objectBoxes[i].width;
        temp.height = objectBoxes[i].height;

        rectangles.push_back(temp);

        personIDs.push_back(objectBoxes[i].objectId);
    }
    return std::make_tuple(rectangles,personIDs);
}