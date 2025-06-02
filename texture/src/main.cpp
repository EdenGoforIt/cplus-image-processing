#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

const int grassLabel = 0;
const int cloudLabel = 1;
const int seaLabel = 2;

// Texture Analysis
// 		IMPLEMENT AND TRAIN A SIMPLE APPROACH TO CLASSIFY TEXTURE WITHIN IMAGES.USING THE SIMPLE CLASSIFIER,
// 		SEGMENT GRASS, CLOUDS AND SEA FROM IMAGES.

// Compute LBP value for a single pixel
uchar calculateLBP(const Mat &img, int x, int y)
{
	uchar center = img.at<uchar>(y, x);
	uchar code = 0;

	// Neighbors positions (clockwise starting from top-left)
	uchar topLeft = img.at<uchar>(y - 1, x - 1);
	uchar top = img.at<uchar>(y - 1, x);
	uchar topRight = img.at<uchar>(y - 1, x + 1);
	uchar right = img.at<uchar>(y, x + 1);
	uchar bottomRight = img.at<uchar>(y + 1, x + 1);
	uchar bottom = img.at<uchar>(y + 1, x);
	uchar bottomLeft = img.at<uchar>(y + 1, x - 1);
	uchar left = img.at<uchar>(y, x - 1);

	// Build a binary code based on the neighbors' intensity compared to the center pixel
	// Then convert into decimal value in the end
	code |= (topLeft > center) << 7;
	code |= (top > center) << 6;
	code |= (topRight > center) << 5;
	code |= (right > center) << 4;
	code |= (bottomRight > center) << 3;
	code |= (bottom > center) << 2;
	code |= (bottomLeft > center) << 1;
	code |= (left > center) << 0;

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
	// Loop through the images in the folder
	for (const auto &entry : std::filesystem::directory_iterator(folder))
	{
		// Make the image grayscale
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
		{
			cerr << "Failed to load image: " << entry.path() << endl;
			continue;
		}

		Mat feature;
		resize(img, feature, Size(64, 64)); // Resize to a fixed size
		Mat hist = computeLBPHistogram(img);
		features.push_back(hist);

		// const int grassLabel = 0;
		// const int cloudLabel = 1;
		// const int seaLabel = 2;
		labels.push_back(label);
	}
}

int main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
		{
			cerr << "Usage: " << argv[0] << " <path_to_test_image>. E.g. ./src/main case1.jpg" << endl;
			return -1;
		}

		Mat features, labels;

		// 1. Load training data (feature extraction + labeling)
		loadTrainingData("../data/grass", grassLabel, features, labels);
		loadTrainingData("../data/cloud", cloudLabel, features, labels);
		loadTrainingData("../data/sea", seaLabel, features, labels);

		cout << "Training data loaded successfully." << endl;

		// 2. Train the k-NN classifier
		Ptr<KNearest> knn = KNearest::create();
		knn->train(features, ROW_SAMPLE, labels);

		logFile << "Classifier trained successfully." << endl;
		cout << "Classifier trained successfully." << endl;

		// 3. Load test image
		string fullInputPath = "../data/" + string(argv[1]);
		Mat testImg = imread(fullInputPath, IMREAD_GRAYSCALE);
		Mat originalImage = imread(fullInputPath, IMREAD_COLOR);

		if (testImg.empty())
		{
			logFile << "Failed to load test image: " << argv[1] << endl;
			cerr << "Failed to load test image: " << argv[1] << endl;
			return -1;
		}
		logFile << "Test image loaded successfully: " << argv[1] << endl;
		cout << "Test image loaded successfully: " << argv[1] << endl;

		// 4. Segment the image patch-by-patch
		Mat result = Mat::zeros(testImg.size(), CV_8UC3);
		int patchSize = 32;
		const int K = 3; // kNN parameter

		for (int y = 0; y < testImg.rows - patchSize; y += patchSize)
		{
			for (int x = 0; x < testImg.cols - patchSize; x += patchSize)
			{
				Mat patch = testImg(Rect(x, y, patchSize, patchSize));
				Mat hist = computeLBPHistogram(patch);

				// 5. Use k-NN to find the K nearest training samples
				Mat neighborResponses, neighborDistances;
				Mat results;
				knn->findNearest(hist, K, results, neighborResponses, neighborDistances);

				// 6. Apply distance-weighted voting
				float weightedVotes[K] = {0};
				for (int i = 0; i < K; ++i)
				{
					float response = neighborResponses.at<float>(0, i);
					float distance = neighborDistances.at<float>(0, i);
					float weight = (distance == 0) ? FLT_MAX : 1.0f / distance;

					if (response == grassLabel)
						weightedVotes[0] += weight;
					else if (response == cloudLabel)
						weightedVotes[1] += weight;
					else if (response == seaLabel)
						weightedVotes[2] += weight;
				}

				// 7. Assign final label and draw rectangle
				int finalClass = max_element(weightedVotes, weightedVotes + 3) - weightedVotes;
				Scalar color;
				// Labels: 0 for grass, 1 for cloud, and 2 for sea
				if (finalClass == grassLabel)
					color = Scalar(0, 255, 0); // Grass as green
				else if (finalClass == cloudLabel)
					color = Scalar(200, 200, 200); // Clouds as grey
				else
					color = Scalar(255, 0, 0); // Sea as blue

				rectangle(result, Rect(x, y, patchSize, patchSize), color, FILLED);
			}
		}

		// Display & save the result
		imshow("Segmented Texture", result);
		imshow("Original Image", originalImage);
		waitKey(0);

		logFile << "Segmentation completed successfully." << endl;
		cout << "Segmentation completed successfully." << endl;

		logFile.close();
	}
	catch (const std::exception &e)
	{
		cerr << "Exception: " << e.what() << endl;
		return -1;
	}

	return 0;
}