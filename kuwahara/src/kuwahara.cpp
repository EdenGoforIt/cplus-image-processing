#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

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

	string outputPath = "../" + std::string(outputPath);
	imwrite(outputPath, outputImage);

	cout << "Successfully implemented Kuwahara Filter. " << endl;
	return 0;
}

void kuwaharaFilter(const Mat &input, Mat &output, int kernelSize)
{
	// Need to check boundary when calculating quadrants
	output = Mat::zeros(input.size(), input.type());
}