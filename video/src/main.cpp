#include <iostream>
#include <opencv2/opencv.hpp>
#include <deque>
#include <fstream>

using namespace cv;
using namespace std;

ofstream logFile("log.txt");

// Generate Gaussian weights for smoothing
vector<double> applyGaussianWeightAverage(int windowSize, double sigma)
{
	vector<double> weights(windowSize);
	double sum = 0.0;
	int center = windowSize / 2;

	for (int i = 0; i < windowSize; ++i)
	{
		int distance = i - center;
		// Gaussian formula
		weights[i] = exp(-(distance * distance) / (2.0 * sigma * sigma));
		sum += weights[i];
	}

	for (double &w : weights)
	{
		// Divide each weight by the total sum to normalize them (so their total becomes 1)
		w /= sum;
	}

	return weights;
}

// Find homography between two frames
Mat findHomographyBetweenFrames(const Mat &previousFrame, const Mat &currentFrame)
{
	Mat defaultH = Mat::eye(3, 3, CV_64F);

	// Convert frames to grayscale as SIFT works only on grayscale images
	Mat prevGray, currGray;
	cvtColor(previousFrame, prevGray, COLOR_BGR2GRAY);
	cvtColor(currentFrame, currGray, COLOR_BGR2GRAY);

	// Feature detection using SIFT which is good for scaled features
	Ptr<SIFT> sift = SIFT::create();
	vector<KeyPoint> kp1, kp2;
	Mat desc1, desc2;

	// Using SIFT detect the keypoitns and descriptor from the frame
	sift->detectAndCompute(prevGray, noArray(), kp1, desc1);
	sift->detectAndCompute(currGray, noArray(), kp2, desc2);

	// For empty descriptors or not enough keypoints, return not transformed homography
	if (desc1.empty() || desc2.empty() || kp1.size() < 10 || kp2.size() < 10)
	{
		logFile << "Not enough features detected" << endl;
		cerr << "Not enough features detected" << endl;
		return defaultH;
	}

	// Match the features
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch>> knnMatches;

	try
	{
		matcher->knnMatch(desc1, desc2, knnMatches, 2);
	}
	catch (cv::Exception &e)
	{
		logFile << "Exception in Matching features: " << e.what() << endl;
		cerr << "Exception in Matching features: " << e.what() << endl;
		return defaultH;
	}

	// Filter matches to find the good ones
	vector<DMatch> goodMatches;
	for (const auto &m : knnMatches)
	{
		if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance)
		{
			goodMatches.push_back(m[0]);
		}
	}
	// If not enough good matches, return just the default homography
	if (goodMatches.size() < 8)
	{
		logFile << "Not enough good matches: " << goodMatches.size() << endl;
		cerr << "Not enough good matches: " << goodMatches.size() << endl;
		return defaultH;
	}

	// Extract matched points
	vector<Point2f> pts1, pts2;
	for (const auto &match : goodMatches)
	{
		pts1.push_back(kp1[match.queryIdx].pt);
		pts2.push_back(kp2[match.trainIdx].pt);
	}

	// Calculate homography
	Mat H = findHomography(pts1, pts2, RANSAC, 3.0);

	if (H.empty())
	{
		logFile << "Failed to find homography" << endl;
		return defaultH;
	}

	// Validate homography
	double det = H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(0, 1) * H.at<double>(1, 0);
	if (fabs(det) < 0.1 || fabs(det) > 10)
	{
		logFile << "Homography determinant out of bounds: " << det << endl;
		return defaultH;
	}

	return H;
}

// Apply Gaussian smoothing to homography matrices
Mat smoothHomographies(const deque<Mat> &matrixBuffer, const vector<double> &weights)
{
	Mat smoothed = Mat::zeros(3, 3, CV_64F);

	// Apply Gaussian average weights to the homography matrices
	for (size_t i = 0; i < matrixBuffer.size(); i++)
	{
		Mat m = matrixBuffer[i].clone();
		double w = weights[i];
		smoothed += w * m;
	}

	// Homographies are always normalized to H(2,2) = 1 by convention
	if (smoothed.at<double>(2, 2) != 0)
	{
		smoothed = smoothed * (1.0 / smoothed.at<double>(2, 2));
	}

	return smoothed;
}

// We are picking up the middle frame as it's the best balanced frame
Mat stabilizeMiddleFrame(const deque<Mat> &frameBuffer,
												 const deque<Mat> &matrixBuffer,
												 const vector<double> &weights,
												 int borderSize, int padding)
{
	int centerIndex = frameBuffer.size() / 2;
	Mat centerFrame = frameBuffer[centerIndex];
	Mat centerH = matrixBuffer[centerIndex];

	Mat smoothedH = smoothHomographies(matrixBuffer, weights);
	Mat correctionH = smoothedH * centerH.inv();

	// Add border and padding
	Mat borderedFrame;
	copyMakeBorder(centerFrame, borderedFrame, borderSize, borderSize, borderSize, borderSize,
								 BORDER_CONSTANT, Scalar(0, 255, 0));

	Mat paddedFrame;
	copyMakeBorder(borderedFrame, paddedFrame, padding, padding, padding, padding,
								 BORDER_CONSTANT, Scalar(0, 255, 0));

	Mat adjustedH = correctionH.clone();
	Mat stabilizedPadded;
	warpPerspective(paddedFrame, stabilizedPadded, adjustedH, paddedFrame.size());

	// Crop to restore original+border size
	int cropX = (paddedFrame.cols - (centerFrame.cols + 2 * borderSize)) / 2;
	int cropY = (paddedFrame.rows - (centerFrame.rows + 2 * borderSize)) / 2;

	Rect cropRect(cropX, cropY,
								centerFrame.cols + 2 * borderSize,
								centerFrame.rows + 2 * borderSize);

	return stabilizedPadded(cropRect).clone();
}

int main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
		{
			logFile << "Usage: ./src/main <video_file>" << endl;
			throw runtime_error("Usage: ./src/main <video_file>");
		}

		// Open video source
		VideoCapture cap;
		cap.open(argv[1]);
		if (!cap.isOpened())
		{
			cerr << "[main] [Error]: Could not open the video file: " << argv[1] << endl;
			cerr << "[main] [Debug]: Make sure file exists and OpenCV has required codecs" << endl;
			return -1;
		}

		// 19 frames (to start with as the requirement) are used to calculate the homography
		const int windowSize = 19;
		const double sigma = 5.0;

		// Smaller padding to make borders more visible
		int padding = 100;
		// Green border size
		int borderSize = 10;

		int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
		logFile << "Total frames in the video: " << totalFrames << endl;
		cout << "Total frames in the video: " << totalFrames << endl;

		// If the frame is less than 19
		if (totalFrames > 0 && totalFrames < windowSize)
		{
			cerr << "Error: Video is too short. Needs at least " << windowSize << " frames." << endl;
			return -1;
		}

		// Calculate Gaussian weights for smoothing
		vector<double> weights = applyGaussianWeightAverage(windowSize, sigma);
		deque<Mat> frameBuffer;
		deque<Mat> matrixBuffer;

		namedWindow("Original", WINDOW_NORMAL);
		namedWindow("Smoothed", WINDOW_NORMAL);

		logFile << "Starting video stabilization..." << endl;

		while (true)
		{
			Mat frame;
			cap >> frame;
			if (frame.empty())
				break;

			frameBuffer.push_back(frame.clone());

			// It only applies when there are at least 2 frames in the buffer
			if (frameBuffer.size() > 1)
			{
				// Calculate homography between previous and current frames
				Mat H = findHomographyBetweenFrames(
						frameBuffer[frameBuffer.size() - 2],
						frameBuffer[frameBuffer.size() - 1]);

				// Save the first homography as the initial cumulative homography
				if (matrixBuffer.empty())
				{
					matrixBuffer.push_back(H.clone());
				}
				else
				{
					// Cumulative homography; new homography from the current frame all the way to the first frame
					matrixBuffer.push_back(H * matrixBuffer.back());
				}
			}
			else
			{
				// if not enough frames, just push the not transformed homography
				matrixBuffer.push_back(Mat::eye(3, 3, CV_64F));
			}

			// Process if the frame buffer has enough frames. Currently 19 frames are used as window size
			if (frameBuffer.size() >= windowSize)
			{
				Mat centerFrame = frameBuffer[windowSize / 2];
				Mat stabilized = stabilizeMiddleFrame(frameBuffer, matrixBuffer, weights, borderSize, padding);

				// Display original and stabilized frames
				imshow("Original", centerFrame);
				imshow("Smoothed", stabilized);

				// Remove oldest frame and matrix
				frameBuffer.pop_front();
				matrixBuffer.pop_front();
			}

			// Exit in any key press
			if (waitKey(0) > 0)
			{
				break;
			}
		}

		cap.release();
		destroyAllWindows();
		logFile << "Video stabilization completed." << endl;
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