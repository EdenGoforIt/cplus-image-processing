#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("../akiyo1.jpg");
	Mat image2;
	image2.create(image.size(), image.channels());

	Point2f center;
	center.x = image.cols / 2;
	center.y = image.rows / 2;

	double angle = 30;
	double scale = 1;

	Mat rotation = getRotationMatrix2D(center, angle, scale);
	warpAffine(image, image2, rotation, image.size());

	imshow("source", image);
	imshow("output", image2);

	waitKey(0);
	return 0;
}