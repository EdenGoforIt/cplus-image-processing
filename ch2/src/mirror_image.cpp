#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("../Binary1.jpg");
	Mat mirrored;

	flip(image, mirrored, 1);
	imshow("Original", image);
	imshow("Mirrored", mirrored);
	waitKey(0);
	return 0;
}
