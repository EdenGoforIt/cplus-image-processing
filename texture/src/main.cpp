#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

ofstream logFile("log.txt");

// Texture Analysis
// 		IMPLEMENT AND TRAIN A SIMPLE APPROACH TO CLASSIFY TEXTURE WITHIN IMAGES.USING THE SIMPLE CLASSIFIER,
// 		SEGMENT GRASS, CLOUDS AND SEA FROM IMAGES.

// Compute LBP value for a single pixel
uchar calculateLBP(const Mat &img, int x, int y)
{
	uchar center = img.at<uchar>(y, x);
	uchar code = 0;
	code |= (img.at<uchar>(y - 1, x - 1) > center) << 7;
	code |= (img.at<uchar>(y - 1, x) > center) << 6;
	code |= (img.at<uchar>(y - 1, x + 1) > center) << 5;
	code |= (img.at<uchar>(y, x + 1) > center) << 4;
	code |= (img.at<uchar>(y + 1, x + 1) > center) << 3;
	code |= (img.at<uchar>(y + 1, x) > center) << 2;
	code |= (img.at<uchar>(y + 1, x - 1) > center) << 1;
	code |= (img.at<uchar>(y, x - 1) > center) << 0;
	return code;
}

// Compute LBP histogram (256-bin) for a patch
Mat computeLBPHistogram(const Mat &patch)
{
	Mat hist = Mat::zeros(1, 256, CV_32F);
	for (int y = 1; y < patch.rows - 1; ++y)
	{
		for (int x = 1; x < patch.cols - 1; ++x)
		{
			uchar lbp = calculateLBP(patch, x, y);
			hist.at<float>(0, lbp)++;
		}
	}
	normalize(hist, hist, 1.0, 0.0, NORM_L1);
	return hist;
}

void loadTrainingData(const string &folder, int label, Mat &features, Mat &labels)
{
	for (const auto &entry : std::filesystem::directory_iterator(folder))
	{
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
		{
			logFile << "Failed to load image: " << entry.path() << endl;
			continue;
		}

		Mat feature;
		resize(img, feature, Size(64, 64)); // Resize to a fixed size
		Mat hist = computeLBPHistogram(img);
		features.push_back(hist);
		labels.push_back(label);
	}
}

int main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
		{
			cerr << "Usage: " << argv[0] << " <path_to_test_image>" << endl;
			return -1;
		}

		Mat features, labels;

		// Prepare training data
		loadTrainingData("../data/grass", 0, features, labels);
		loadTrainingData("../data/cloud", 1, features, labels);
		loadTrainingData("../data/sea", 2, features, labels);

		logFile << "Training data loaded successfully." << endl;
		cout << "Training data loaded successfully." << endl;

		// Train Classifier
		Ptr<KNearest> knn = KNearest::create();
		knn->train(features, ROW_SAMPLE, labels);

		logFile << "Classifier trained successfully." << endl;
		cout << "Classifier trained successfully." << endl;

		// Load test image using command line argument
		string fullInputPath = "../data/" + string(argv[1]);
		Mat testImg = imread(fullInputPath, IMREAD_GRAYSCALE);

		if (testImg.empty())
		{
			logFile << "Failed to load test image: " << argv[1] << endl;
			cerr << "Failed to load test image: " << argv[1] << endl;
			return -1;
		}
		logFile << "Test image loaded successfully: " << argv[1] << endl;
		cout << "Test image loaded successfully: " << argv[1] << endl;

		// Compute LBP histogram for the test image
		Mat result = Mat::zeros(testImg.size(), CV_8UC3);
		int patchSize = 32;

		for (int y = 0; y < testImg.rows - patchSize; y += patchSize)
		{
			for (int x = 0; x < testImg.cols - patchSize; x += patchSize)
			{
				Mat patch = testImg(Rect(x, y, patchSize, patchSize));
				Mat hist = computeLBPHistogram(patch);
				float response = knn->predict(hist);

				Scalar color;
				if (response == 0)
					color = Scalar(0, 255, 0); // Grass
				else if (response == 1)
					color = Scalar(200, 200, 200); // Clouds
				else
					color = Scalar(255, 0, 0); // Sea

				rectangle(result, Rect(x, y, patchSize, patchSize), color, FILLED);
			}
		}

		// Display & save the result
		imshow("Segmented Texture", result);
		imshow("Original Image", testImg);
		waitKey(0);

		logFile << "Segmentation completed successfully." << endl;
		cout << "Segmentation completed successfully." << endl;

		logFile.close();
	}
	catch (const std::exception &e)
	{
		logFile << "Exception: " << e.what() << endl;
		cerr << "Exception: " << e.what() << endl;
		logFile.close();
		return -1;
	}

	return 0;
}