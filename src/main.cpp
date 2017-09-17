// Standard include files
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
 
using namespace cv;
using namespace std;
 
int main(int argc, char **argv)
{
    // Set up tracker. 
    // Instead of MIL, you can also use 
    // BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN  
    Ptr<TrackerTLD> tracker = TrackerTLD::create();
 
    // Read video
    VideoCapture video(0);
     
    // Check video is open
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }
 
    // Read first frame. 
    Mat frame;
    video.read(frame);
 
    // Define an initial bounding box
    Rect2d bbox(287, 23, 86, 320);
 
    // Uncomment the line below if you 
    // want to choose the bounding box
    // bbox = selectROI(frame, false);
     
    // Initialize tracker with first frame and bounding box
    tracker->init(frame, bbox);
 
    while(video.read(frame))
    {
        // Update tracking results
        tracker->update(frame, bbox);
 
        // Draw bounding box
        rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
 
        // Display result
        imshow("Tracking", frame);
        unsigned char key = waitKey(1);
        if (key == 'x')
        {
            // if user presses "x" then exit
            std::cout << "Keyboard exit requested : exiting now - bye!" << std::endl;
            break;
        }
 
    }
 
    return 0; 
     
}