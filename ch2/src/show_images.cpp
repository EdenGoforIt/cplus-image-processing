#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	// Read image
	Mat_<Vec3b> image1 = imread("../Binary1.jpg");
	Mat_<Vec3b> image2 = imread("../akiyo1.jpg");

	// Named window
	namedWindow("image1", 0);
	namedWindow("image2", 0);

	imshow("image1", image1);
	waitKey(0);
	imshow("image2", image2);
	waitKey(0);
}