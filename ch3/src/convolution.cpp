#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class ImageConvolution
{
private:
	Mat kernel;
	int kernelSize;

public:
	explicit ImageConvolution(int size) : kernelSize(size)
	{
		if (kernelSize % 2 == 0)
		{
			kernelSize++;
		}
		createKernel();
	}

	void createKernel()
	{
		kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / static_cast<float>(kernelSize * kernelSize);
	}

	Mat apply(const Mat &input)
	{
		Mat output;
		filter2D(input, output, -1, kernel);
		return output;
	}

	// Getter
	int getKernelSize() const { return kernelSize; }
	Mat getKernel() const { return kernel; }
};

int main()
{
	Mat input = imread("../akiyo1.jpg", IMREAD_GRAYSCALE);

	ImageConvolution convolution(10);

	Mat output = convolution.apply(input);

	imshow("original", input);
	imshow("output", output);

	waitKey(0);
	return 0;
}