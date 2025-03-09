#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat img1, img2;
int defaultValue = 0;

void callback_trackbar(int click, void *object)
{
	cout << "click: " << click << endl;
	defaultValue = click;
	if (defaultValue == 0)
	{
		imshow("show", img1);
	}
	else
	{
		imshow("show", img2);
	}
}

int main()
{
	img1 = imread("../akiyo1.jpg");
	cvtColor(img1, img2, COLOR_BGR2GRAY);

	// Create window first
	namedWindow("show");

	createTrackbar("trackBar", "show", &defaultValue, 1, callback_trackbar);

	setTrackbarPos("trackBar", "show", 0);

	imshow("show", img1);
	waitKey(0);
	return 0;
}