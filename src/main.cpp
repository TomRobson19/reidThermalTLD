#include "person.hpp"
#include "tracker.hpp"

using namespace cv;
using namespace std;

#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"

int main(int argc, char **argv)
{
    // Read video
    VideoCapture video("data/Dataset1/betaInput.webm");

    // Check video is open
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int width = 30;
    int height = 30;
    int learning = 100000;
    int padding = 40;


    Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500, 25, false);

    CascadeClassifier cascade = CascadeClassifier(CASCADE_TO_USE);

    Mat frame;

    MultiObjectTLDTracker tracker = MultiObjectTLDTracker();

    int personCounter = 0;
    
    while(video.read(frame))
    {  
        tracker.update(frame);

        Mat displayImage = frame.clone();

        tracker.drawBoxes(displayImage);

        Mat foreground;
        MoG->apply(frame, foreground, (double)(1.0 / learning));

        // perform erosion - removes boundaries of foreground object
        erode(foreground, foreground, Mat(), Point(), 1);

        // perform morphological closing
        dilate(foreground, foreground, Mat(), Point(), 5);
        erode(foreground, foreground, Mat(), Point(), 1);

        // get connected components from the foreground
        findContours(foreground, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        //make new image from bitwise and of frame and foreground

        for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
        {
            Rect r = boundingRect(contours[idx]);

            // adjust bounding rectangle to be padding% larger
            // around the object
            r.x = max(0, r.x - (int) (padding / 100.0 * (double) r.width));
            r.y = max(0, r.y - (int) (padding / 100.0 * (double) r.height));

            r.width = min(frame.cols - 1, (r.width + 2 * (int) (padding / 100.0 * (double) r.width)));
            r.height = min(frame.rows - 1, (r.height + 2 * (int) (padding / 100.0 * (double) r.height)));

            // draw rectangle if greater than width/height constraints and if
            // also still inside image
            if ((r.width >= width) && (r.height >= height) && (r.x + r.width < frame.cols) && (r.y + r.height < frame.rows))
            {
                vector<Rect> found, found_filtered;

                rectangle(displayImage, r, Scalar(0, 0, 255), 2, 1 );

                Mat roi = frame(r);

                auto temp = tracker.getObjectRectangles();

                std::vector<Rect> objectRectangles = std::get<0>(temp);
                std::vector<int> personIDs = std::get<1>(temp);

                bool alreadyTarget = false;

                for (int i = 0; i < objectRectangles.size(); i++)
                {
                    if ((r & objectRectangles[i]).area() > 0) 
                    {
                        alreadyTarget = true;
                    }
                }

                if (!alreadyTarget)
                {
                    cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(32, 64));

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
                    }

                    for (size_t i = 0; i < found_filtered.size(); i++)
                    {
                        Rect rec = found_filtered[i];
                        rectangle(displayImage, rec, Scalar(0, 255, 0), 2, 1 );

                        int key = waitKey(10000000);
                        cout << "####" << key << "####" << personCounter << endl;
                        tracker.addTarget(rec, personCounter);
                        personCounter++;   
                    }
                    
                }
            }
        }

        auto temp = tracker.getObjectRectangles();

        std::vector<Rect> objectRectangles = std::get<0>(temp);
        std::vector<int> personIDs = std::get<1>(temp);

        for (int i = 0; i < objectRectangles.size(); i++)
        {
            if (objectRectangles[i].x<20 || objectRectangles[i].x+objectRectangles[i].width>630 || objectRectangles[i].y<0 || objectRectangles[i].y+objectRectangles[i].height>480)
            {
                cout << "deletion" << personIDs[i] << endl;
                tracker.deleteTarget(personIDs[i]);

                temp = tracker.getObjectRectangles();
                objectRectangles = std::get<0>(temp);
                personIDs = std::get<1>(temp);
            }
            else
            {
                Mat roi = frame(objectRectangles[i]);
                vector<Rect> found, found_filtered;

                cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(32, 64));

                for(size_t i = 0; i < found.size(); i++ )
                {
                    Rect rec = found[i];

                    rec.x += objectRectangles[i].x;
                    rec.y += objectRectangles[i].y;

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
                }
                cout << found_filtered.size() << endl;
            }
        }

        // Display result
        imshow("Tracking", displayImage);
        unsigned char key = waitKey(10);
        if (key == 'x')
        {
            // if user presses "x" then exit
            std::cout << "Keyboard exit requested : exiting now - bye!" << std::endl;
            break;
        }
    }
    return 0;
}