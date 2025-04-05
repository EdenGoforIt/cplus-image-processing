#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

// Debug
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int gridSize = 47;
ofstream logFile("log.txt");

// Color map for 8 colors to avoid magic strings
struct ColorMap
{
	const Vec3b black = {0, 0, 0};
	const Vec3b red = {255, 0, 0};
	const Vec3b green = {0, 255, 0};
	const Vec3b blue = {0, 0, 255};
	const Vec3b yellow = {255, 255, 0};
	const Vec3b magenta = {255, 0, 255};
	const Vec3b cyan = {0, 255, 255};
	const Vec3b white = {255, 255, 255};
};
ColorMap colorMap;
struct Vec3bComparator
{
	bool operator()(const Vec3b &a, const Vec3b &b) const
	{
		if (a[0] != b[0])
			return a[0] < b[0];
		if (a[1] != b[1])
			return a[1] < b[1];
		return a[2] < b[2];
	}
};
// Encdoing Array table. First chracter is space.
char encodingArray[64] = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'w', 'z',
													'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'W', 'Z',
													'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'};

// Simple 8 color code Hash map
map<Vec3b, string, Vec3bComparator> eightColorMap = {
		{colorMap.black, "000"},	 // Black
		{colorMap.red, "100"},		 // Red
		{colorMap.green, "010"},	 // Green
		{colorMap.blue, "001"},		 // Blue
		{colorMap.yellow, "110"},	 // Yellow
		{colorMap.magenta, "101"}, // Magenta
		{colorMap.cyan, "011"},		 // Cyan
		{colorMap.white, "111"}		 // White
};

/// @brief The marker zone is the 6x6 square in the top-left, bottom-left, and bottom-right corners of the barcode.
/// @param row Grid row
/// @param col Grid column
/// @return if the square is in the marker zone
bool isInMarkerZone(int row, int col)
{
	// 6x6 markers excluded in 47x47 grid
	return (row < 6 && col < 6) ||	 // top-left marker
				 (row >= 41 && col < 6) || // bottom-left marker
				 (row >= 41 && col >= 41); // bottom-right marker
}

/// @brief The decoding is composed of 8 colors code, we need to find the closest color
/// @param pixel  The pixel color to find the closest color for
/// @return The closest color in the colorMap
Vec3b findClosestColor(Vec3b pixel)
{
	Vec3b closestColor = colorMap.black;
	int minimumDistance = INT_MAX;

	for (const auto &color : eightColorMap)
	{
		int distance = norm(pixel - color.first);
		if (distance < minimumDistance)
		{
			minimumDistance = distance;
			closestColor = color.first;
		}
	}

	return closestColor;
}

/// @brief Detect the area of the barcode in the image. This is necessary as the square is bordered with black with some width
/// @param image The input image to detect the barcode area
/// @return The bounding rectangle of the detected barcode area
Rect detectBarcodeArea(const Mat &image)
{
	Mat gray, thresh;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, thresh, 220, 255, THRESH_BINARY_INV);
	// Use morphological operations to close gaps in the barcode
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);
	vector<Point> points;
	findNonZero(thresh, points);
	// Return empty rectangle if no white pixels are found
	if (points.empty())
	{
		return Rect();
	}

	Rect bbox = boundingRect(points);
	return bbox;
}

/// @brief Decode the barcode from the provided image.
/// @param image Image
/// @return Decoded string
/// @note Not removing debug code code as accuracy is more important than performance
string decodeBarcode(const Mat &image)
{
	string decoded;

	// Check the offset of the image. Because the image is filled with black border, we need to exclude rough border width
	Rect roughBorderRectangle = detectBarcodeArea(image);
	double squareWidth = roughBorderRectangle.width / static_cast<double>(gridSize);
	double squareHeight = roughBorderRectangle.height / static_cast<double>(gridSize);
	double offsetX = roughBorderRectangle.x;
	double offsetY = roughBorderRectangle.y;

	vector<string> bitsList;

	// Debug
	logFile << "[decodeBarcode] [Debug]: Image size: " << image.cols << "x" << image.rows << endl;
	logFile << "[decodeBarcode] [Debug]: Border Rectangle: x=" << roughBorderRectangle.x << ", y=" << roughBorderRectangle.y << ", w=" << roughBorderRectangle.width << ", h=" << roughBorderRectangle.height << endl;
	logFile << "[decodeBarcode] [Debug]: Square size: " << squareWidth << "x" << squareHeight << endl;
	logFile << "[decodeBarcode] [Debug]: Offset: " << offsetX << ", " << offsetY << endl;

	// Debug
	Mat debugImg = image.clone();

	for (int y = 0; y < gridSize; y++)
	{
		for (int x = 0; x < gridSize; x++)
		{
			// 6x6 markers excluded in 47x47 grid; one top-left, one bottom-left, and one bottom-right
			if (isInMarkerZone(y, x))
			{
				continue;
			}

			double x0 = offsetX + x * squareWidth;
			double x1 = offsetX + (x + 1) * squareWidth;
			double y0 = offsetY + y * squareHeight;
			double y1 = offsetY + (y + 1) * squareHeight;
			Point center(round((x0 + x1) / 2.0), round((y0 + y1) / 2.0));

			// Debug
			circle(debugImg, center, 1, Scalar(0, 0, 255), FILLED);

			Vec3b pixel = image.at<Vec3b>(center);
			// Quantize the pixel color to the closest color in the 8 color map
			Vec3b quantized = findClosestColor(pixel);
			// Convert the appropriate color to a string of bits
			string bits = eightColorMap[quantized];

			// Debug
			logFile << "[decodeBarcode] [DEBUG]: square (" << x << "," << y << ") center: " << center << endl;
			logFile << "[decodeBarcode] [DEBUG]: pixel: (" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ")\n";
			logFile << "[decodeBarcode] [DEBUG]: quantized: (" << (int)quantized[0] << "," << (int)quantized[1] << "," << (int)quantized[2] << ")\n";
			logFile << "[decodeBarcode] [DEBUG]: bits: " << bits << endl;

			bitsList.push_back(bits);
		}
	}

	// Combine the bits into a single string
	for (size_t i = 0; i + 1 < bitsList.size(); i += 2)
	{
		// Bit is composed of two colors
		string bits = bitsList[i] + bitsList[i + 1];
		// Convert the 6 bits to a number
		int index = bitset<6>(bits).to_ulong();
		// 2^6 = 64. Double check the index
		if (index < 64)
		{
			decoded += encodingArray[index];
		}
	}

	logFile << "[decodeBarcode] [DEBUG]: Decoded string = " << decoded << endl;

	// Debug
	// Check if the center is calculated right. Check debug_center.jpg in ./build/debug_centeres.jpg
	imwrite("debug_centers.jpg", debugImg);
	return decoded;
}

/// @brief Align the barcode image to the correct orientation using three corners; top-left, bottom-left, and bottom-right.
/// @param image Original image
/// @return	Aligned image
Mat alignBarcodeImage(const Mat &image)
{
	// Remove noise and convert to grayscale to find the corners
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(9, 9), 2);

	// Check if there are three circles in the image
	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 5, 100, 35, 20, 40);
	logFile << "[Align Image] [Debug]: Found circles: " << circles.size() << endl;

	if (circles.size() != 3)
	{
		logFile << "[Align Image] [Error]: Could not find three circles in the image" << endl;
		throw invalid_argument("[Align Image] [Error]: Could not find three circles in the image");
	}

	return image;
}

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
		const char *inputPath = argv[1];
		string fullInputPath = "../" + string(inputPath);
		Mat inputImage = imread(fullInputPath, IMREAD_COLOR);

		if (inputImage.empty())
		{
			cerr << "[main] [Error]: Could not open or find the image: " << fullInputPath << endl;
			return -1;
		}

		// Based on the assumption that the image is full size without any white padding or border
		Mat alignedImage = alignBarcodeImage(inputImage);
		string decoded = decodeBarcode(alignedImage);

		if (decoded.empty())
		{
			cerr << "[main] [Error]: Could not decode the barcode" << endl;
			return -1;
		}

		cout << "[main] [Debug] Decoded barcode: " << decoded << endl;

		cout << "[main] [Debug] Successfully processed the barcode" << endl;

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
		cerr << "[main] [Error]: OpenCV Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception &e)
	{
		cerr << "[main] Error: Standard Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		cerr << "[main] [Error]: An unknown error occurred" << std::endl;
		return -1;
	}
}