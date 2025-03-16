#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

void kuwaharaFilter(const Mat &input, Mat &output, int kernelSize)
{
	// Create output image
	output = Mat(input.size(), input.type());
	int halfKernalSize = kernelSize / 2;

	// Create integral images (summed-area tables)
	Mat sumImage, sqSumImage;

	// Convert to double for precision; using int used to cause overflow and blurred images
	Mat inputDouble;
	input.convertTo(inputDouble, CV_64F);

	// Calculate integral images
	integral(inputDouble, sumImage, sqSumImage);

	// Process each pixel
	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			// Arrays to store region info
			double regionMean[4] = {0};
			double regionVar[4] = {0};
			int regionSize[4] = {0};

			// Process the 4 regions (quadrants) around current pixel
			// Region 1: Top-Left
			int r1_y1 = max(0, y - halfKernalSize);
			int r1_x1 = max(0, x - halfKernalSize);
			int r1_y2 = y;
			int r1_x2 = x;

			// Region 2: Top-Right
			int r2_y1 = max(0, y - halfKernalSize);
			int r2_x1 = x;
			int r2_y2 = y;
			int r2_x2 = min(input.cols, x + halfKernalSize + 1);

			// Region 3: Bottom-Left
			int r3_y1 = y;
			int r3_x1 = max(0, x - halfKernalSize);
			int r3_y2 = min(input.rows, y + halfKernalSize + 1);
			int r3_x2 = x;

			// Region 4: Bottom-Right
			int r4_y1 = y;
			int r4_x1 = x;
			int r4_y2 = min(input.rows, y + halfKernalSize + 1);
			int r4_x2 = min(input.cols, x + halfKernalSize + 1);

			// Calculate region sizes
			regionSize[0] = (r1_y2 - r1_y1) * (r1_x2 - r1_x1);
			regionSize[1] = (r2_y2 - r2_y1) * (r2_x2 - r2_x1);
			regionSize[2] = (r3_y2 - r3_y1) * (r3_x2 - r3_x1);
			regionSize[3] = (r4_y2 - r4_y1) * (r4_x2 - r4_x1);

			// Region coordinates array for easier processing
			int regions[4][4] = {
					{r1_y1, r1_x1, r1_y2, r1_x2},
					{r2_y1, r2_x1, r2_y2, r2_x2},
					{r3_y1, r3_x1, r3_y2, r3_x2},
					{r4_y1, r4_x1, r4_y2, r4_x2}};

			// Calculate mean and variance for each region using integral images
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

					// Get sum from integral image
					double sum = sumImage.at<double>(y2, x2) - sumImage.at<double>(y2, x1) -
											 sumImage.at<double>(y1, x2) + sumImage.at<double>(y1, x1);

					// Get sum of squares from integral image
					double sqSum = sqSumImage.at<double>(y2, x2) - sqSumImage.at<double>(y2, x1) -
												 sqSumImage.at<double>(y1, x2) + sqSumImage.at<double>(y1, x1);

					// Calculate mean and variance
					regionMean[i] = sum / regionSize[i];
					regionVar[i] = (sqSum / regionSize[i]) - (regionMean[i] * regionMean[i]);

					// Keep track of minimum variance
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

/**
 * @brief Main Function to apply Kuwahara filter
 * @param argc Number of Arguments
 * @param argv Arguments
 * @return int Return Status
 */
int main(int argc, char **argv)
{
	// Argument Validation
	if (argc != 4)
	{
		cerr << "!! Wrong Arguments." << argv[0] << " <input> <output> <kernel_size>. e.g. ./src/kuwahara limes_kuwahara5x5.tif output1.jpg 5" << endl;
		return -1;
	}

	// Argument Conversion
	const char *inputPath = argv[1];
	const char *outputPath = argv[2];
	int kernelSize = atoi(argv[3]);

	cout << "Converted - Input: " << inputPath << ". Output: " << outputPath << ". Kernel Size: " << kernelSize << endl;

	// Kernel Size Validation
	if (kernelSize % 2 == 0 || kernelSize < 3 || kernelSize > 15)
	{
		cerr << "!! Kernel Size should be odd and between 3 and 15." << endl;
		return -1;
	}

	// Read Image from Input Path and as Gray Scale according to the insturciton
	// File is located under the parent or root directory
	string fullInputPath = "../" + std::string(inputPath);
	Mat inputImage = imread(fullInputPath, IMREAD_GRAYSCALE);

	if (inputImage.empty())
	{
		cerr << "!!Input Image cannot be read." << endl;
		return -1;
	}
	// Timer starts here for perfomance measurement
	auto startTime = high_resolution_clock ::now();

	Mat outputImage;
	kuwaharaFilter(inputImage, outputImage, kernelSize);

	// Measure the time taken to apply the Kuwahara Filter
	auto stopTime = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stopTime - startTime);
	cout << "Processing time: " << duration.count() << " milliseconds" << endl;

	string fullOutputPath = "../" + std::string(outputPath);
	imwrite(fullOutputPath, outputImage);

	cout << "Successfully implemented Kuwahara Filter. " << endl;
	return 0;
}
