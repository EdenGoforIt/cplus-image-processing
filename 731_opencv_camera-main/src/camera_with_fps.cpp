#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace chrono;


int main(int argc, char** argv) {
    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    cap.open(0);
    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    cout << "Opened camera" << endl;
    namedWindow("WebCam", 1);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
    cap >> frame;
    printf("frame size %d %d\n", frame.rows, frame.cols);
    int key = 0;
    float fps = 0.0;
    while (1) {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if (frame.empty()) break;
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 255), 2, 8);
        imshow("WebCam", frame);
        key = waitKey(1);
        if (key == 113 || key == 27 || cv::getWindowProperty("WebCam", cv::WND_PROP_VISIBLE) < 1) {return 0; } // either esc or 'q' or close window
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
        cout << "frames " << fps << frame(0, 0) << endl;
    }
}
