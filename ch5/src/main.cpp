#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

int main()
{

	Mat image = imread("../akiyo1.jpg");
	if (image.empty())
	{
		cerr << "Error: Could not open image file." << endl;
		return -1;
	}

		waitKey(0);
	return 0;
}