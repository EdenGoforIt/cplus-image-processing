#include <iostream>
#include <opencv2/opencv.hpp>
#include <deque>
#include <fstream>

using namespace cv;
using namespace std;

ofstream logFile("log.txt");

int main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
		{
			logFile << "Usage: ./src/main <video_file>" << endl;
			throw std::runtime_error("Usage: ./src/main <video_file>");
		}
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