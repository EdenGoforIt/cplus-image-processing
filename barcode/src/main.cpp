#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

// Debug
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

char encodingArray[64] = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'w', 'z',
													'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'W', 'Z',
													'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'};

int main(int argc, char **argv)
{
	// TODO: match with the arguments for now let's read
	if (argc != 2)
	{
		cerr << "!! Wrong Arguments. " << argv[0] << " <input image file name> e.g. ./src/main 2DEmpty.jpg" << endl;
		return -1;
	}

	try
	{
		const char *inputPath = argv[1];

		cout << "Arguments - Input: " << inputPath << endl;

		

		cout << "Successfully read the barcode" << endl;
		return 0;
	}
	catch (const std::invalid_argument &e)
	{
		cerr << "Invalid argument Exception: " << e.what() << "\n";
		return -1;
	}
	catch (const cv::Exception &e)
	{
		cerr << "OpenCV Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception &e)
	{
		cerr << "Standard Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		cerr << "An unknown error occurred" << std::endl;
		return -1;
	}
}