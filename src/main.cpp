#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
 
using namespace cv;
using namespace std;

#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"

int main(int argc, char **argv)
{
    Ptr<MultiTrackerTLD> allTrackers();

    Ptr<TrackerTLD> tracker = TrackerTLD::create();
 
    // Read video
    VideoCapture video("data/Dataset1/betaInput.webm");
     
    // Check video is open
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }
 
    // Set up Mat for frame. 
    Mat frame;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int width = 40;
    int height = 100;
    int learning = 1000;
    int padding = 40; 


    Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500,25,false);

    CascadeClassifier cascade = CascadeClassifier(CASCADE_TO_USE);


    //THIS NEEDS TO BE FOR EACH NEW DETECTION
    // Define an initial bounding box
    Rect2d bbox;
     
    // Initialize tracker with first frame and bounding box
    //tracker->init(frame, bbox);
 
    while(video.read(frame))
    {
        Mat foreground;
        MoG->apply(frame, foreground, (double)(1.0/learning));

        // perform erosion - removes boundaries of foreground object
        erode(foreground, foreground, Mat(),Point(),1);

        // perform morphological closing
        dilate(foreground, foreground, Mat(),Point(),5);
        erode(foreground, foreground, Mat(),Point(),1);

        // get connected components from the foreground
        findContours(foreground, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        imshow("foreground",foreground);


        for(int idx = 0; idx >=0; idx = hierarchy[idx][0])
        {
            Rect r = boundingRect(contours[idx]);

            // adjust bounding rectangle to be padding% larger
            // around the object
            r.x = max(0, r.x - (int) (padding/100.0 * (double) r.width));
            r.y = max(0, r.y - (int) (padding/100.0 * (double) r.height));

            r.width = min(frame.cols - 1, (r.width + 2 * (int) (padding/100.0 * (double) r.width)));
            r.height = min(frame.rows - 1, (r.height + 2 * (int) (padding/100.0 * (double) r.height)));

            // draw rectangle if greater than width/height constraints and if
            // also still inside image
            if ((r.width >= width) && (r.height >= height) && (r.x + r.width < frame.cols) && (r.y + r.height < frame.rows))
            {
                vector<Rect> found, found_filtered;

                Mat roi = frame(r);

                cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(32,64));

                for(size_t i = 0; i < found.size(); i++ )
                {
                    Rect rec = found[i];

                    rec.x += r.x;
                    rec.y += r.y;

                    size_t j;
                    // Do not add small detections inside a bigger detection.
                    for ( j = 0; j < found.size(); j++ )
                    {
                      if ( j != i && (rec & found[j]) == rec )
                      {
                          break;
                      }
                    }

                    if (j == found.size())
                    {
                      found_filtered.push_back(rec);
                    }

                    rectangle(frame, rec.tl(), rec.br(), (255,0,0) , 3);

                    tracker->init(frame, rec);
                }
            }
        }

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