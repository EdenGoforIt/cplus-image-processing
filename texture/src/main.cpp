#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace cv::ml;
using namespace std;

ofstream logFile("log.txt");

const int grassLabel = 0;
const int cloudLabel = 1;
const int seaLabel = 2;
const int K = 3; // kNN parameter

int confusionMatrix[3][3] = {0}; // Confusion matrix [True][Predicted]

uchar calculateLBP(const Mat &img, int x, int y)
{
	uchar center = img.at<uchar>(y, x);
	uchar code = 0;

	uchar topLeft = img.at<uchar>(y - 1, x - 1);
	uchar top = img.at<uchar>(y - 1, x);
	uchar topRight = img.at<uchar>(y - 1, x + 1);
	uchar right = img.at<uchar>(y, x + 1);
	uchar bottomRight = img.at<uchar>(y + 1, x + 1);
	uchar bottom = img.at<uchar>(y + 1, x);
	uchar bottomLeft = img.at<uchar>(y + 1, x - 1);
	uchar left = img.at<uchar>(y, x - 1);

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
	for (const auto &entry : filesystem::directory_iterator(folder))
	{
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
		{
			logFile << "Failed to load image: " << entry.path() << endl;
			continue;
		}
		Mat hist = computeLBPHistogram(img);
		features.push_back(hist);
		labels.push_back(label);
	}
}

void printConfusionMatrix()
{
	cout << "\n+-----------+-------+-------+-------+" << endl;
	cout << "| Predicted | Grass | Cloud | Sea   |" << endl;
	cout << "+-----------+-------+-------+-------+" << endl;
	string classes[3] = {"Grass    ", "Cloud    ", "Sea      "};
	for (int i = 0; i < 3; ++i)
	{
		cout << "| " << classes[i] << "|";
		for (int j = 0; j < 3; ++j)
		{
			cout << "   " << confusionMatrix[i][j] << "  |";
		}
		cout << endl;
	}
	cout << "+-----------+-------+-------+-------+\n"
			 << endl;
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
		loadTrainingData("../data/grass", grassLabel, features, labels);
		loadTrainingData("../data/cloud", cloudLabel, features, labels);
		loadTrainingData("../data/sea", seaLabel, features, labels);

		Ptr<KNearest> knn = KNearest::create();
		knn->setDefaultK(K);
		knn->train(features, ROW_SAMPLE, labels);

		string fullInputPath = "../data/" + string(argv[1]);
		Mat testImg = imread(fullInputPath, IMREAD_GRAYSCALE);

		if (testImg.empty())
		{
			cerr << "Failed to load test image: " << argv[1] << endl;
			return -1;
		}

		Mat result = Mat::zeros(testImg.size(), CV_8UC3);
		int patchSize = 32;

		for (int y = 0; y < testImg.rows - patchSize; y += patchSize)
		{
			for (int x = 0; x < testImg.cols - patchSize; x += patchSize)
			{
				Mat patch = testImg(Rect(x, y, patchSize, patchSize));
				Mat hist = computeLBPHistogram(patch);

				Mat neighborResponses, neighborDistances;
				Mat results;
				knn->findNearest(hist, K, results, neighborResponses, neighborDistances);

				float weightedVotes[3] = {0};
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

				int finalClass = max_element(weightedVotes, weightedVotes + 3) - weightedVotes;

				// Dummy true label for demonstration (you should replace this with actual ground truth)
				int trueLabel = grassLabel; // Replace this accordingly based on the actual test data
				confusionMatrix[trueLabel][finalClass]++;

				Scalar color;
				if (finalClass == grassLabel)
					color = Scalar(0, 255, 0); // Grass as green
				else if (finalClass == cloudLabel)
					color = Scalar(200, 200, 200); // Clouds as grey
				else
					color = Scalar(255, 0, 0); // Sea as blue

				rectangle(result, Rect(x, y, patchSize, patchSize), color, FILLED);
			}
		}

		imshow("Segmented Texture", result);
		imshow("Original Image", testImg);
		waitKey(0);

		// Print confusion matrix after segmentation
		printConfusionMatrix();

		logFile.close();
	}
	catch (const std::exception &e)
	{
		cerr << "Exception: " << e.what() << endl;
		logFile.close();
		return -1;
	}

	return 0;
}