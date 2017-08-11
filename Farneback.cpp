#include <iostream>  
#include "opencv2/opencv.hpp"  
  
using namespace cv;  
using namespace std;  
  
#define UNKNOWN_FLOW_THRESH 1e9  
  
// Color encoding of flow vectors from:  
// http://members.shaw.ca/quadibloc/other/colint.htm  
// This code is modified from:  
// http://vision.middlebury.edu/flow/data/  

void motionToColor(Mat flow, Mat &color)  
{  
    Mat xy[2];
    split(flow, xy);

    //calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0/mag_max);

    //build hsv image
    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    //convert to BGR and show
    // Mat bgr;//CV_32FC3 matrix
    cvtColor(hsv, color, COLOR_HSV2BGR);
    // imshow("flow", bgr);
}  
  
int main(int, char**)  
{  
    VideoCapture cap;  
    cap.open(0);  
    //cap.open("test_02.wmv");  
  
    if( !cap.isOpened() )  
        return -1;  
  
    Mat prevgray, gray, flow, cflow, frame;  
    namedWindow("flow", 1);  
  
    Mat motion2color;  
  
    for(;;)  
    {  
        double t = (double)cvGetTickCount();  
  
        cap >> frame;  
        cvtColor(frame, gray, CV_BGR2GRAY);  
        imshow("original", frame);  
  
        if( prevgray.data )  
        {  
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
            motionToColor(flow, motion2color);  
                        // By y += 5, x += 5 you can specify the grid 
            for (int y = 0; y < frame.rows; y += 5) 
            {
                for (int x = 0; x < frame.cols; x += 5)
                {
                  // get the flow from y, x position * 10 for better visibility
                    const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
                             // draw line at flow direction
                    line(frame, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
                                                                 // draw initial point
                    circle(frame, Point(x, y), 1, Scalar(0, 0, 0), -1);
                }
            }
            imshow("flow", motion2color);
            imshow("flow vector", frame);
            // imshow("flow", flow);
        }  
        if(waitKey(1) == 27)
        {
            printf("exit");
            break;
        }
        std::swap(prevgray, gray);  
  
        t = (double)cvGetTickCount() - t;  
        cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;  
    }  
    return 0;  
}  
