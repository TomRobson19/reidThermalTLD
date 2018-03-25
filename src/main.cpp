#include "tracker.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <string.h>

using namespace cv;
using namespace std;

#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"

#define imagesFIFO "/tmp/images.fifo" 
#define intsFIFO "/tmp/ints.fifo" 
#define imagesDirectory "/tmp/imgs/"

int fileNameCounter = 0;

void writeToFIFO(Mat img) { 
    cout << "writing" << endl;
    int num, fifo; 
    char extension[]=".jpg"; 
    char newLine[]="\n"; 

    std::string name = std::to_string(fileNameCounter);
    char nameChar[name.length()+1]; 
    
    strcpy(nameChar, name.c_str());

    char fileName[256]; // <- danger, only storage for 256 characters.
    strncpy(fileName, imagesDirectory, sizeof(fileName));
    strncat(fileName, nameChar, sizeof(fileName));
    strncat(fileName, extension, sizeof(fileName));
    

    imwrite(fileName, img);

    //strncat(fileName, newLine, sizeof(fileName));

    cout << "here" << endl;

    fifo = open(imagesFIFO, O_WRONLY);

    cout << fileName << endl;

    num= write(fifo, fileName, strlen(fileName));

    close(fifo);
    fileNameCounter++;
}

int readFromFIFO(){ 
    int num, fifo, status; 
    char temp[32]; 

    fifo = open(intsFIFO, O_RDONLY);

    num = read(fifo, temp, sizeof(temp));
    
    close(fifo);

    cout << temp << endl;

    return stoi(temp);

}   

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
    int width = 100;
    int height = 100;
    int learning = 100000;
    int padding = 40;


    Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500, 25, false);

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    Mat frame;

    MultiObjectTLDTracker tracker = MultiObjectTLDTracker();

    while(video.read(frame))
    {  
        resize(frame, frame, Size(1280, 960));
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

                //rectangle(displayImage, r, Scalar(0, 0, 255), 2, 1 );

                Mat roi = frame(r);

                vector<rectangleAndID> objectRectangles = tracker.getObjectRectangles();

                bool alreadyTarget = false;

                for (int i = 0; i < objectRectangles.size(); i++)
                {
                    if ((r & objectRectangles[i].rectangle).area() > 0) 
                    {
                        alreadyTarget = true;
                    }
                }

                if (!alreadyTarget)
                {
                    hog.detectMultiScale(roi, found, 0, Size(8,8), Size(8,16), 1.05, 2);

                    for(size_t i = 0; i < found.size(); i++ )
                    {
                        Rect rec = found[i];

                        rec.x += r.x;
                        rec.y += r.y;

                        bool save = true;
                        for (int j = 0; j < found.size(); j++ )
                        {
                            Rect currentComparison = found[j];
                            currentComparison.x += r.x;
                            currentComparison.y += r.y;

                            Rect combination = rec & currentComparison;

                            if ( ((combination.area() > 0) || (combination.area() == rec.area())) && i != j )
                            {
                                save = false;
                                break;
                            }
                        }
                        if (save == true)
                        {
                            found_filtered.push_back(rec);
                        }
                    }

                    for (int i = 0; i < found_filtered.size(); i++)
                    {
                        Rect rec = found_filtered[i];
                        rectangle(displayImage, rec, Scalar(0, 255, 0), 2, 1 );

                        //image to be sent to the neural network
                        Mat imgToUse = frame(rec);
                        resize(imgToUse, imgToUse, Size(256,512));

                        //Neural Network Placeholder
                        writeToFIFO(imgToUse);

                        int personID = readFromFIFO();

                        Point2f center = Point2f(float(rec.x + rec.width/2.0), float(rec.y + rec.height/2.0));

                        char str[200];
                        sprintf(str,"Person %d",personID);

                        putText(displayImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

                        //don't use the resized version here!
                        tracker.addTarget(rec, personID);
                    }
                    
                }
            }
        }

        std::vector<rectangleAndID> objectRectangles = tracker.getObjectRectangles();

        for (int i = 0; i < objectRectangles.size(); i++)
        {
            if (objectRectangles[i].rectangle.x<0 || objectRectangles[i].rectangle.x+objectRectangles[i].rectangle.width>1280 || objectRectangles[i].rectangle.y<0 || objectRectangles[i].rectangle.y+objectRectangles[i].rectangle.height>960)
            {
                cout << "deletion border" << objectRectangles[i].personID << endl;
                tracker.deleteTarget(objectRectangles[i].personID);
                cout << "deleted" << endl;

                std::vector<rectangleAndID> objectRectangles = tracker.getObjectRectangles();

                cout << objectRectangles.size() << endl;
            }
            else
            {
                Rect r = objectRectangles[i].rectangle;

                Mat roi = frame(r);
                vector<Rect> found, found_filtered;

                hog.detectMultiScale(roi, found, 0, Size(8,8), Size(8,16), 1.05, 2);

                for(int j = 0; j < found.size(); j++ )
                {
                    Rect rec = found[j];

                    rec.x += objectRectangles[i].rectangle.x;
                    rec.y += objectRectangles[i].rectangle.y;

                    int k;
                    // Do not add small detections inside a bigger detection.
                    for ( k = 0; k < found.size(); k++ )
                    {
                        if (((rec & found[k]).area() > 0) && (found[k].area() < rec.area()))
                        {
                            break;
                        }
                    }

                    if (k == found.size())
                    {
                        found_filtered.push_back(rec);
                    }
                }
                if (found_filtered.size() == 0)
                {
                    cout << "deletion hog" << objectRectangles[i].personID << endl;
                    tracker.deleteTarget(objectRectangles[i].personID);
                    cout << "deleted" << endl;

                    std::vector<rectangleAndID> objectRectangles = tracker.getObjectRectangles();
                    cout << objectRectangles.size() << endl;
                }
                else if (found_filtered[0].area()*2 < objectRectangles[i].rectangle.area())
                {
                    cout << "deletion size" << objectRectangles[i].personID << endl;
                    tracker.deleteTarget(objectRectangles[i].personID);
                    cout << "deleted" << endl;

                    std::vector<rectangleAndID> objectRectangles = tracker.getObjectRectangles();
                    cout << objectRectangles.size() << endl;
                }
                else
                {
                    Rect rec = found_filtered[i];
                    Point2f center = Point2f(float(rec.x + rec.width/2.0), float(rec.y + rec.height/2.0));

                    int personID = objectRectangles[i].personID;

                    char str[200];
                    sprintf(str,"Person %d",personID);

                    putText(displayImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));
                }
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