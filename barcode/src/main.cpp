#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

// Debug
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// First chracter is space
char encodingArray[64] = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'w', 'z',
													'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'W', 'Z',
													'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'};

// Simple 8 color code Hash map
map<Vec3b, string> colorMap = {
		{{0, 0, 0}, "000"},			 // Black
		{{255, 0, 0}, "100"},		 // Red
		{{0, 255, 0}, "010"},		 // Green
		{{0, 0, 255}, "001"},		 // Blue
		{{255, 255, 0}, "110"},	 // Yellow
		{{255, 0, 255}, "101"},	 // Magenta
		{{0, 255, 255}, "011"},	 // Cyan
		{{255, 255, 255}, "111"} // White
};

string decodeBarcode(const Mat &inputImage)
{
	// Suppress unused parameter warning
	(void)inputImage;

	return "placeholder_value";
}

int main(int argc, char **argv)
{
	// Validation; Argument should have two parameters.
	if (argc != 2)
	{
		cerr << "!! Wrong Arguments. " << argv[0] << " <input image file name> e.g. ./src/main 2DEmpty.jpg" << endl;
		return -1;
	}

	try
	{
		const char *inputPath = argv[1];
		string fullInputPath = "../" + string(inputPath);
		Mat inputImage = imread(fullInputPath, IMREAD_GRAYSCALE);

		if (inputImage.empty())
		{
			cerr << "!! Error: Could not open or find the image: " << fullInputPath << endl;
			return -1;
		}

		// TODO: check rotated images

		string decoded = decodeBarcode(inputImage);

		if (decoded.empty())
		{
			cerr << "!! Error: Could not decode the barcode" << endl;
			return -1;
		}

		cout << "Decoded barcode: " << decoded << endl;

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