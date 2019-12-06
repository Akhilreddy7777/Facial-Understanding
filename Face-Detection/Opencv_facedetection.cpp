#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;
ofstream myfile;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

#define CAFFE

const std::string caffeConfigFile = "deploy.prototxt";
const std::string caffeWeightFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "opencv_face_detector_uint8.pb";

vector<Rect> detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, string file1)
{
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;
	std::vector<Rect> faces;
#ifdef CAFFE
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif
	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	int q = 0;
	cout << "detection mat rows is " << detectionMat.rows << std::endl;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		q = q + 1;
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
			myfile << file1 <<","<< x1 << "," << y1 << "," << x2 << "," << y2 << std::endl;
			//myfile << x1, y1, x2, y2;
			cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
			int width = x2 - x1;
			int height = y2 - y1;
			cv::Rect inp = Rect(x1, y1, width, height);
			faces.push_back(inp);
			

		}

		//cout << "i inside loop is" << q << std::endl;
	}
	return faces;

}


int main(int argc, const char** argv)
{
#ifdef CAFFE
	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
	Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile);
#endif

	vector<String> filenames;
	String folder = "./faces/";
	glob(folder, filenames);
	myfile.open("opencv_dnn_FDDB.csv");
	vector<Rect> faces;
	int q = 0;
	for (size_t i = 0; i < filenames.size(); i++)
	{
		q = q + 1;
		Mat frame = imread(filenames[i]);
		if (frame.empty())
		{
			std::cerr << "Can't read image from the file: " << filenames[i] << std::endl;
			exit(-1);
		}

		cout << "Channels: " + to_string(frame.channels()) << endl;

		double tt_opencvDNN = 0;
		double fpsOpencvDNN = 0;

		faces = detectFaceOpenCVDNN(net, frame, filenames[i]);
		//imshow("OpenCV - DNN Face Detection", frame);

		int k = waitKey(0);

	}
	cout << "i is " << q << std::endl;
	myfile.close();


}

