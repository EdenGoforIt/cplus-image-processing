#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

/// @brief Calculate Integral Image and Sqaure Integraal Image. (Summed Area Tables (SAT))
/// @details SAT tables will be used to calculate the mean and variances of regions in Kuwahara Filter
/// The details of the algorithm can be found in https://en.wikipedia.org/wiki/Summed-area_table#:~:text=In%20the%20image%20processing%20domain,Crow%20for%20use%20with%20mipmaps.
/// @param input input Image to calculate Sum and Square Sum images
/// @param sumImage output Sum Image
/// @param sqSumImage output Square Sum Image
void calculateIntegralImages(const Mat &input, Mat &sumImage, Mat &sqSumImage)
{
	// When creating the integral image, it is convenient to create a matrix (or IplImage)
	// with one extra column and one extra row, both initialised to zero. This is to avoid when x or y becomes -1
	sumImage = Mat::zeros(input.rows + 1, input.cols + 1, CV_64F);
	sqSumImage = Mat::zeros(input.rows + 1, input.cols + 1, CV_64F);

	// Fill integral images row by row. Iterate the height (rows) first
	for (int y = 0; y < input.rows; y++)
	{
		// Iterate the width (cols) of the image
		for (int x = 0; x < input.cols; x++)
		{
			// Get pixel value (as double for precision)
			double pixel = input.at<uchar>(y, x);
			double pixelSq = pixel * pixel;

			// Calculate sums (+1 because integral images have extra row/col)
			// I(x,y) = i(x,y) + I(x-1, y) + I(x,y -1) +  - I(x-1, y-1). In Matrix, element is accessed by matrix[row, col] = matrix [y,x]
			sumImage.at<double>(y + 1, x + 1) = pixel +
																					sumImage.at<double>(y, x + 1) +
																					sumImage.at<double>(y + 1, x) -
																					sumImage.at<double>(y, x);

			// Sqaure Sum will be used to calculate variance
			sqSumImage.at<double>(y + 1, x + 1) = pixelSq +
																						sqSumImage.at<double>(y, x + 1) +
																						sqSumImage.at<double>(y + 1, x) -
																						sqSumImage.at<double>(y, x);
		}
	}
}

void kuwaharaFilter(const Mat &input, Mat &output, int kernelSize)
{
	// Create output image
	output = Mat(input.size(), input.type());
	int halfKernelSize = kernelSize / 2;

	// Calculate integral images before to be used without complexiing the code
	// According to the book, it might be safer to convert to double. Or we can calculate the max by max = width*height*255
	Mat sumImage, sqSumImage;
	Mat inputDoubleImage;
	input.convertTo(inputDoubleImage, CV_64F);

	// Calculate integral images already to improve performance
	calculateIntegralImages(inputDoubleImage, sumImage, sqSumImage);

	// Process each pixel
	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			// Arrays to store region info
			double regionMean[4] = {0};
			double regionVar[4] = {0};
			int regionSize[4] = {0};

			// Define the 4 regions (quadrants) around current pixel

			// Region 1: Top-Left (A)
			int r1_y1 = max(0, y - halfKernelSize);
			int r1_y2 = y;
			int r1_x1 = max(0, x - halfKernelSize);
			int r1_x2 = x;

			// Region 2: Top-Right
			int r2_y1 = max(0, y - halfKernelSize);							 // 0
			int r2_y2 = y;																			 // 0
			int r2_x1 = x;																			 // 1
			int r2_x2 = min(input.cols - 1, x + halfKernelSize); // -1 as index starts with 0 // 6

			// Region 3: Bottom-Left
			int r3_y1 = y;
			int r3_y2 = min(input.rows - 1, y + halfKernelSize); // Changed to -1
			int r3_x1 = max(0, x - halfKernelSize);
			int r3_x2 = x;

			// Region 4: Bottom-Right
			int r4_y1 = y;
			int r4_y2 = min(input.rows - 1, y + halfKernelSize); // Changed to -1
			int r4_x1 = x;
			int r4_x2 = min(input.cols - 1, x + halfKernelSize); // Changed to -1

			// Calculate region sizes
			// Adding + 1 to 3 - 1 becomes 2 but it should be 3 pixels
			regionSize[0] = (r1_y2 - r1_y1 + 1) * (r1_x2 - r1_x1 + 1);
			regionSize[1] = (r2_y2 - r2_y1 + 1) * (r2_x2 - r2_x1 + 1);
			regionSize[2] = (r3_y2 - r3_y1 + 1) * (r3_x2 - r3_x1 + 1);
			regionSize[3] = (r4_y2 - r4_y1 + 1) * (r4_x2 - r4_x1 + 1);

			// Region coordinates array
			int regions[4][4] = {
					{r1_y1, r1_x1, r1_y2, r1_x2},
					{r2_y1, r2_x1, r2_y2, r2_x2},
					{r3_y1, r3_x1, r3_y2, r3_x2},
					{r4_y1, r4_x1, r4_y2, r4_x2}};

			// Calculate mean and variance for each region
			double minVar = DBL_MAX;
			int minVarIndex = 0;

			for (int i = 0; i < 4; i++)
			{
				if (regionSize[i] > 0)
				{
					int y1 = regions[i][0];
					int x1 = regions[i][1];
					int y2 = regions[i][2];
					int x2 = regions[i][3];

					// Get sum from integral image with boundary checking
					double bottomRight = sumImage.at<double>(y2, x2);
					double bottomLeft = (x1 > 0) ? sumImage.at<double>(y2, x1 - 1) : 0;
					double topRight = (y1 > 0) ? sumImage.at<double>(y1 - 1, x2) : 0;
					double topLeft = (y1 > 0 && x1 > 0) ? sumImage.at<double>(y1 - 1, x1 - 1) : 0;

					// Area = pt4 - pt2 - pt3 + pt1
					double sum = bottomRight - bottomLeft - topRight + topLeft;

					// Get sum of squares with boundary checking
					double sqBottomRight = sqSumImage.at<double>(y2, x2);
					double sqBottomLeft = (x1 > 0) ? sqSumImage.at<double>(y2, x1 - 1) : 0;
					double sqTopRight = (y1 > 0) ? sqSumImage.at<double>(y1 - 1, x2) : 0;
					double sqTopLeft = (y1 > 0 && x1 > 0) ? sqSumImage.at<double>(y1 - 1, x1 - 1) : 0;

					double sqSum = sqBottomRight - sqBottomLeft - sqTopRight + sqTopLeft;

					// Calculate mean and variance
					regionMean[i] = sum / regionSize[i];
					regionVar[i] = (sqSum / regionSize[i]) - (regionMean[i] * regionMean[i]);

					if (regionVar[i] < minVar)
					{
						minVar = regionVar[i];
						minVarIndex = i;
					}
				}
			}

			// Set output pixel to mean of region with minimum variance
			output.at<uchar>(y, x) = saturate_cast<uchar>(regionMean[minVarIndex]);
		}
	}
}

int main(int argc, char **argv)
{
	// Argument Validation. Argument should have three parameters.
	if (argc != 4)
	{
		cerr << "!! Wrong Arguments. " << argv[0] << " <input> <output> <kernel_size>. e.g. ./src/kuwahara limes_kuwahara5x5.tif output1.jpg 5" << endl;
		return -1;
	}

	// Argument Conversion
	const char *inputPath = argv[1];
	const char *outputPath = argv[2];
	int kernelSize = atoi(argv[3]);

	cout << "Converted - Input: " << inputPath << ". Output: " << outputPath << ". Kernel Size: " << kernelSize << endl;

	// Kernel Size Validation. Odd nubmer works better.
	if (kernelSize % 2 == 0 || kernelSize < 3 || kernelSize > 15)
	{
		cerr << "!! Kernel Size should be odd and between 3 and 15." << endl;
		return -1;
	}

	// Read Image. The image should be under the root directory. Otherwise change this path.
	string fullInputPath = "../" + string(inputPath);
	Mat inputImage = imread(fullInputPath, IMREAD_GRAYSCALE);

	if (inputImage.empty())
	{
		cerr << "!! Input Image cannot be read." << endl;
		return -1;
	}

	// Timer starts
	auto startTime = high_resolution_clock::now();

	Mat outputImage;
	kuwaharaFilter(inputImage, outputImage, kernelSize);

	// Measure time
	auto stopTime = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stopTime - startTime);
	cout << "Processing time: " << duration.count() / 1000.0 << " milliseconds" << endl;

	// The image should be under the root directory. Otherwise change this path.
	string fullOutputPath = "../" + string(outputPath);
	imwrite(fullOutputPath, outputImage);

	cout << "Successfully implemented Kuwahara Filter." << endl;
	return 0;
}