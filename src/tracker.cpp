#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

class GenericTracker
{
public:
    GenericTracker(VideoCapture capture)
    {}
};

class MultiObjectTLDTracker : public GenericTracker
{
public:
    MultiObjectTLDTracker(VideoCapture capture) : GenericTracker(capture)
    {}
};