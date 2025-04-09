#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{
	Mat input = imread("../peppers.bmp");

	// Color segmentation
	// Mat output = Mat::zeros(input.size(), input.type());

	// for (int y = 0; y < input.rows; y++)
	// {
	// 	for (int x = 0; x < input.cols; x++)
	// 	{
	// 		Vec3b pixel = input.at<Vec3b>(y, x);
	// 		int g = pixel[1];
	// 		int r = pixel[2];
	// 		Vec3b &out = output.at<Vec3b>(y, x);

	// 		if (r > 240 && g < 200)
	// 		{ // Orange
	// 			out = Vec3b(0, 128, 255);
	// 		}
	// 		else if (r > 200 && g > 200)
	// 		{ // Yellow
	// 			out = Vec3b(0, 255, 255);
	// 		}
	// 		else if (g > 100 && r < 100)
	// 		{ // Green
	// 			out = Vec3b(0, 255, 0);
	// 		}
	// 		else if (r > 100 && g < 100)
	// 		{ // Red
	// 			out = Vec3b(0, 0, 255);
	// 		}
	// 	}
	// }

	// imshow("Original", input);
	// imshow("RGB Segmentation", output);

	// HSV to RGB
	// Mat hsv;
	// cvtColor(input, hsv, COLOR_BGR2HSV);

	// vector<Mat> channels(3);
	// split(hsv, channels);
	// imshow("Hue", channels[0]);
	// imshow("Saturation", channels[1]);
	// imshow("Value", channels[2]);

	// Mat classified = Mat::zeros(input.size(), input.type());
	// for (int y = 0; y < hsv.rows; y++)
	// {
	// 	for (int x = 0; x < hsv.cols; x++)
	// 	{
	// 		Vec3b pixel = hsv.at<Vec3b>(y, x);
	// 		int h = pixel[0];
	// 		int s = pixel[1];

	// 		Vec3b &out = classified.at<Vec3b>(y, x);
	// 		if (s < 15)
	// 		{
	// 			out = Vec3b(0, 0, 0); // black
	// 		}
	// 		else if (h < 10 || h > 160)
	// 		{
	// 			out = Vec3b(0, 0, 255); // Red
	// 		}
	// 		else if (h >= 10 && h < 25)
	// 		{ // Orange
	// 			out = Vec3b(0, 128, 255);
	// 		}
	// 		else if (h >= 25 && h < 45)
	// 		{ // Yellow
	// 			out = Vec3b(0, 255, 255);
	// 		}
	// 		else if (h >= 45 && h < 65)
	// 		{ // Green
	// 			out = Vec3b(0, 255, 0);
	// 		}
	// 	}
	// }

	// imshow("HSV Classification", classified);

	// RGB to HSV

	// RGB
	// It's about showing the intensity of rgb into the input; if blue colors are intense, it shows as white
	// Mat input = imread("../akiyo1.jpg");
	// vector<Mat> channels(3);
	// split(input, channels);

	// imshow("original", input);
	// imshow("Blue", channels[0]);
	// imshow("Green", channels[1]);
	// imshow("Red", channels[2]);

	waitKey(0);
	return 0;
}