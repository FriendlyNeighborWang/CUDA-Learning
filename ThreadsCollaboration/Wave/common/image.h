#ifndef __IMAGE__
#define __IMAGE__

#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


class IMAGE
{
public:
    cv::Mat image;


    IMAGE( int w, int h) 
    {
        image = cv::Mat::zeros(w,h,CV_8UC4);
        imshow("images",image);
    }


    unsigned char* get_ptr( void ) const   
    { 
        return (unsigned char*)image.data; 
    }

    long image_size( void ) const 
    { 
		return image.cols * image.rows * 4; 
    }


    char show_image(int time=0)
    {
        imshow("images",image);
        return cv::waitKey(time);
    }

};



#endif  // __IMAGE__

