#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

// Debug
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int gridSize = 47;

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
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);
	vector<Point> points;
	findNonZero(thresh, points);
	if (points.empty())
		return Rect(); // Return empty rectangle if no white pixels are found
	Rect bbox = boundingRect(points);
	return bbox;
}

string decodeBarcode(const Mat &image)
{
	ofstream logFile("log.txt");
	string decoded;

	// Check the offset of the image. Because the image is filled with black border, we need to exclude rough border width
	Rect roi = detectBarcodeArea(image);
	double squareWidth = roi.width / static_cast<double>(gridSize);
	double squareHeight = roi.height / static_cast<double>(gridSize);
	double offsetX = roi.x;
	double offsetY = roi.y;

	vector<string> bitsList;

	// Debug
	logFile << "Image size: " << image.cols << "x" << image.rows << endl;
	logFile << "ROI: x=" << roi.x << ", y=" << roi.y << ", w=" << roi.width << ", h=" << roi.height << endl;
	logFile << "Square size: " << squareWidth << "x" << squareHeight << endl;
	logFile << "Offset: " << offsetX << ", " << offsetY << endl;

	Mat debugImg = image.clone();
	for (int y = 0; y < gridSize; y++)
	{
		for (int x = 0; x < gridSize; x++)
		{
			// 6x6 markers excluded in 47x47 grid
			if (isInMarkerZone(y, x))
			{
				continue;
			}

			double x0 = offsetX + x * squareWidth;
			double x1 = offsetX + (x + 1) * squareWidth;
			double y0 = offsetY + y * squareHeight;
			double y1 = offsetY + (y + 1) * squareHeight;
			Point center(round((x0 + x1) / 2.0), round((y0 + y1) / 2.0));
			circle(debugImg, center, 1, Scalar(0, 0, 255), FILLED);

			Vec3b pixel = image.at<Vec3b>(center);
			// Quantize the pixel color to the closest color in the 8 color map
			Vec3b quantized = findClosestColor(pixel);
			// Convert the appropriate color to a string of bits
			string bits = eightColorMap[quantized];

			// Debug
			logFile << "[DEBUG] square (" << x << "," << y << ") center: " << center << endl;
			logFile << "  pixel: (" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ")\n";
			logFile << "  quantized: (" << (int)quantized[0] << "," << (int)quantized[1] << "," << (int)quantized[2] << ")\n";
			logFile << "  bits: " << bits << endl;

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
	logFile << "Decoded string = " << decoded << endl;

	// Check if the center is calculated right. Check debug_center.jpg in ./build/debug_centeres.jpg
	imwrite("debug_centers.jpg", debugImg);
	return decoded;
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
		Mat inputImage = imread(fullInputPath, IMREAD_COLOR);

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