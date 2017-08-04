#include <iostream>  
#include "opencv2/opencv.hpp" 
#include <stdio.h>
  
using namespace cv;  
using namespace std;  

  
int main(int, char**)  
{  
    VideoCapture cap;  
    cap.open(0);  
    //cap.open("test_02.wmv");  
  
    if( !cap.isOpened() )  
        return -1;  
  
    vector<Point2f> prev_corner;
    // vector <Point2f> prev_corner_good, cur_corner_good;

    Mat gray, prevgray, frame;


  // maxCorners – The maximum number of corners to return. If there are more corners
  // than that will be found, the strongest of them will be returned
    int maxCorners = 200;

  // qualityLevel – Characterizes the minimal accepted quality of image corners;
  // the value of the parameter is multiplied by the by the best corner quality
  // measure (which is the min eigenvalue, see cornerMinEigenVal() ,
  // or the Harris function response, see cornerHarris() ).
  // The corners, which quality measure is less than the product, will be rejected.
  // For example, if the best corner has the quality measure = 1500,
  // and the qualityLevel=0.01 , then all the corners which quality measure is
  // less than 15 will be rejected.
    double qualityLevel = 0.02;

  // minDistance – The minimum possible Euclidean distance between the returned corners
    double minDistance = 20.;

    namedWindow("corners", 1);
    namedWindow("flow", 1); 
  
    Mat motion2color; 
    RNG rng(12345);
    int line_thickness = 3;

    cap >> frame;  
    cvtColor(frame, prevgray, CV_BGR2GRAY);
    goodFeaturesToTrack( prevgray, prev_corner, maxCorners, qualityLevel, minDistance); 
    for( size_t i = 0; i < prev_corner.size(); i++ )
    {
        cv::circle( prevgray, prev_corner[i], 5, cv::Scalar( 255. ), -1 );
    }
    imshow("corners", prevgray);

    Mat mask = Mat::zeros(frame.size(), frame.type());
  
    vector<Point2f> cur_corner(prev_corner);
    for(;;)  
    {  
        double t = (double)cvGetTickCount();  
  
        cap >> frame;  
        cvtColor(frame, gray, CV_BGR2GRAY);  
        imshow("original", frame);  

        vector<Point2f> temp_corner(cur_corner);

        if( prevgray.data )  
        {  
            vector <uchar> status;
            vector <float> err;
            calcOpticalFlowPyrLK(prevgray, gray, prev_corner, cur_corner, status, err);
            // weed out bad matches
            for (size_t i = 0; i < status.size(); i++) {

                if (status[i]) {
                    line( mask, temp_corner[i], cur_corner[i], Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), line_thickness);
                    // prev_corner_good.push_back(prev_corner[i]);
                    // cur_corner_good.push_back(cur_corner[i]);
                }
            }
            // motionToColor(flow, motion2color);  
            // imshow("flow", motion2color);
            frame = frame + mask;
            imshow("flow", frame);
        }
        if(waitKey(30) == 27)
        {
            printf("exit");
            break;
        }
  
        t = (double)cvGetTickCount() - t;  
        cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;  
    }  
    return 0;  
}  
