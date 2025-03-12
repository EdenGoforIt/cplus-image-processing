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

	Mat rotation = (Mat_<double>(2, 3) << 0, 0, 0, 0, 0, 0);

	rotation.at<double>(0, 0) = 0.866;
	rotation.at<double>(0, 1) = 0.5;
	rotation.at<double>(0, 2) = -24.210;
	rotation.at<double>(1, 0) = -0.5;
	rotation.at<double>(1, 1) = 0.866;
	rotation.at<double>(1, 2) = 53.646;

	cout << rotation << endl;

	warpAffine(image, image2, rotation, image.size());
	imshow("image", image);
	imshow("image2", image2);
	waitKey(0);
	return 0;
}