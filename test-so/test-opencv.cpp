#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    Mat image = Mat::zeros(100,100,CV_8UC3);
    imshow("1", image);
    waitKey(0);
}