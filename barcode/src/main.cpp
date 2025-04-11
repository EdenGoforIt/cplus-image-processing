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
		{
			return a[0] < b[0];
		}
		if (a[1] != b[1])
		{
			return a[1] < b[1];
		}

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
/// @return return true if the square is in the marker zone, otherwise false
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
		Vec3i pixelSigned = static_cast<Vec3i>(pixel);
		Vec3i colorSigned = static_cast<Vec3i>(color.first);

		// Check how those two colors are far apart
		int distance = norm(pixelSigned - colorSigned);

		// If found the minimum distance, update the closest color
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
	logFile << endl;
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
			// Add a small dot to the image so that we can see center of the square is calculated right
			// Which can be seen in debug image debug_centers.jpg under 'build' folder
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
			logFile << endl;

			bitsList.push_back(bits);
		}
	}

	// Debug big list
	logFile << "[decodeBarcode] [DEBUG]: Bits list: " << endl;
	for (const auto &bits : bitsList)
	{
		logFile << bits << " ";
	}
	logFile << endl;

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

	logFile << "[decodeBarcode] [DEBUG]: Decoded string: " << endl;
	logFile << decoded << endl;
	logFile << endl;
	cout << decoded << endl;

	// Debug
	imwrite("debug_centers.jpg", debugImg);
	return decoded;
}

/// @brief Align the image if the image is rotated
/// @param image Input image
/// @return return aligned image
Mat alignBarcodeImage(const Mat &image)
{
	// Detect blue color for more accuracy
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(100, 150, 50), Scalar(140, 255, 255), mask);

	// Blur to reduce noise and improve circle detection
	GaussianBlur(mask, mask, Size(9, 9), 2);

	// Find the circle using Hough Circles
	vector<Vec3f> circles;
	HoughCircles(mask, circles, HOUGH_GRADIENT, 1, mask.rows / 5, 100, 35, 20, 40);
	if (circles.size() != 3)
	{
		logFile << "[Align Image] [Error]: Found " << circles.size() << " circles, expected 3" << endl;
		throw invalid_argument("[Align Image] [Error]: Could not find exactly three circles");
	}

	// Extract centers only; c[2] will be radius
	vector<Point2f> centers;
	for (const auto &c : circles)
	{
		centers.emplace_back(c[0], c[1]);
	}

	// Determine right-angle triangle (bottom-left marker should be the right angle)
	Point2f A = centers[0], B = centers[1], C = centers[2];
	Point2f rightAngle, topLeft, bottomRight;
	double distanceBetweenAB = norm(A - B);
	double distanceBetweenAC = norm(A - C);
	double distanceBetweenBC = norm(B - C);

	// Find the right angle using Pythagorean Theorem a^2 + b^2 = c^2
	auto isRightTriangle = [](double a2, double b2, double c2)
	{
		double sum = a2 + b2;
		double diff = abs(sum - c2);
		return diff <= c2 * 0.05;
	};

	if (isRightTriangle(distanceBetweenAB * distanceBetweenAB, distanceBetweenAC * distanceBetweenAC, distanceBetweenBC * distanceBetweenBC))
	{
		rightAngle = A;
		topLeft = B;
		bottomRight = C;
	}
	else if (isRightTriangle(distanceBetweenAB * distanceBetweenAB, distanceBetweenBC * distanceBetweenBC, distanceBetweenAC * distanceBetweenAC))
	{
		rightAngle = B;
		topLeft = A;
		bottomRight = C;
	}
	else if (isRightTriangle(distanceBetweenAC * distanceBetweenAC, distanceBetweenBC * distanceBetweenBC, distanceBetweenAB * distanceBetweenAB))
	{
		rightAngle = C;
		topLeft = A;
		bottomRight = B;
	}
	else
	{
		logFile << "[Align Image] [Error]: No right-angle triangle found" << endl;
		throw invalid_argument("[Align Image] [Error]: No right-angle triangle found");
	}

	// Explicitly assign top-left and bottom-right based on coordinates
	// Top-left should have the smallest y-coordinate
	// Bottom-right should have the largest x-coordinate
	// This is to avoid the flip of the image
	if (topLeft.y > bottomRight.y)
	{
		swap(topLeft, bottomRight);
	}
	if (topLeft.x > bottomRight.x)
	{
		swap(topLeft, bottomRight);
	}

	// Define source and destination points
	vector<Point2f> src = {topLeft, rightAngle, bottomRight};

	// Set destination points to a full-size square
	// This canvas and region is based on the examples images
	float targetCanvas = 1200.0f;
	float barcodeRegion = 940.0f;
	float padding = (targetCanvas - barcodeRegion) / 2.0f;

	vector<Point2f> dst = {
			Point2f(padding, padding),																// Top-left
			Point2f(padding, padding + barcodeRegion),								// Bottom-left
			Point2f(padding + barcodeRegion, padding + barcodeRegion) // Bottom-right
	};

	// Rotate based on the calculation
	Mat affine = getAffineTransform(src, dst);
	Mat aligned;
	warpAffine(image, aligned, affine, Size(targetCanvas, targetCanvas), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

	// Debug: Aligned image can be seen in debug-rotated.jpg under 'build' folder
	imwrite("debug-rotated.jpg", aligned);
	return aligned;
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
		Mat inputImage = imread(inputPath, IMREAD_COLOR);

		if (inputImage.empty())
		{
			cerr << "[main] [Error]: Could not open or find the image: " << inputPath << endl;
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