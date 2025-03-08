#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("../akiyo1.jpg");
	Mat image2;
	image2.create(image.size(), CV_8U);

	cvtColor(image, image2, COLOR_BGR2GRAY);
	imwrite("output.png", image2);

	waitKey(0);
}