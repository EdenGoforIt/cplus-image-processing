#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("../akiyo1.jpg", IMREAD_COLOR);
	int kernelSize = 5;
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / float(kernelSize * kernelSize);

	Mat smoothed;
	filter2D(gray, smoothed, -1, kernel);

	imshow("image", image);
	imshow("image2", smoothed);
	waitKey(0);
	return 0;
}