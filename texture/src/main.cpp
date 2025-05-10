#include <iostream>
#include <opencv2/opencv.hpp>
#include <deque>
#include <fstream>

using namespace cv;
using namespace std;

ofstream logFile("log.txt");

// Texture Analysis
// 		IMPLEMENT AND TRAIN A SIMPLE APPROACH TO CLASSIFY TEXTURE WITHIN IMAGES.USING THE SIMPLE CLASSIFIER,
// 		SEGMENT GRASS, CLOUDS AND SEA FROM IMAGES.

void loadTrainingImages(const string &path, int label, Mat &features, Mat &labels)
{
	

}

int main(int argc, char **argv)
{
	try
	{
		Mat features, labels;

		cout << "Opening video file: " << argv[1] << endl;
		logFile << "Video stabilization completed." << endl;
		logFile.close();
	}
	catch (const std::exception &e)
	{
		logFile << "Exception: " << e.what() << endl;
		cerr << "Exception: " << e.what() << endl;
		logFile.close();
		return -1;
	}

	return 0;
}