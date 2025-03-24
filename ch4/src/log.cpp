#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	Mat input = imread("../akiyo1.jpg");

	Mat LoG5x5 = (Mat_<float>(5, 5) << 1, 3, 4, 3, 1,
								3, 0, -6, 0, 3,
								4, -6, -20, -6, 4,
								3, 0, -6, 0, 3,
								1, 3, 4, 3, 1);

	Mat LaplacianKernel = (Mat_<float>(3, 3) << 1, 1, 1,
												 1, -8, 1,
												 1, 1, 1);

	Mat GaussianKernel = (Mat_<float>(3, 3) << 1, 2, 1,
												2, 4, 2,
												1, 2, 1) /
											 16.0;

	Mat imageLoG1, imageLoG2;

	// apply LOG directly
	filter2D(input, imageLoG1, -1, LoG5x5, Point(-1, -1), 0, BORDER_DEFAULT);

	// Gaussian and Laplacian
	Mat blurred;
	filter2D(input, blurred, -1, GaussianKernel, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(blurred, imageLoG2, -1, LaplacianKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	// Nomalize the image
	normalize(imageLoG1, imageLoG1, 0, 255, NORM_MINMAX);
	normalize(imageLoG2, imageLoG2, 0, 255, NORM_MINMAX);

	imageLoG1.convertTo(imageLoG1, CV_8UC1);
	imageLoG2.convertTo(imageLoG2, CV_8UC1);
	imshow("original", input);
	imshow("LOG Direct", imageLoG1);
	imshow("Gaussian + Laplacian", imageLoG2);
	waitKey(0);
	return 0;
}