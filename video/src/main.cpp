#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

// Debug
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

ofstream logFile("log.txt");

int main(int argc, char **argv)
{
	// Validation; Argument should have two parameters.
	if (argc != 2)
	{
		cerr << "[main] [Error]: " << argv[0] << " <input image file name> e.g. ./src/main 2DEmpty.jpg" << endl;
		return -1;
	}

	try
	{

		cout << "[main] [Debug] Successfully processed the barcode" << endl;

		// Debug
		logFile.close();

		return 0;
	}
	catch (const std::invalid_argument &e)
	{
		cerr << "[main] [Error]: Invalid argument Exception: " << e.what() << "\n";
		return -1;
	}
	catch (const cv::Exception &e)
	{
		cerr << "[main] [Error]: OpenCV Exception: " << e.what() << endl;
		return -1;
	}
	catch (const std::exception &e)
	{
		cerr << "[main] Error: Standard Exception: " << e.what() << endl;
		return -1;
	}
	catch (...)
	{
		cerr << "[main] [Error]: An unknown error occurred" << endl;
		return -1;
	}
}