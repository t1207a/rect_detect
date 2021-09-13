#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int lower_r = 40, upper_r = 80;
int lower_g = 40, upper_g = 80;
int lower_b = 40, upper_b = 80;
Mat frame, cvt_frame, mask;

void on_hue_changed(int, void*);
void Label(Mat& img, const vector<Point>& pts, const String& label);

int main(int argc, char* argv[])
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "camera open failed!" << endl;
		return 0;
	}
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

	
		//namedWindow("mask");
		createTrackbar("Lower r", "mask", &lower_r, 255, on_hue_changed);
		//createTrackbar("Upper r", "mask", &upper_r, 255, on_hue_changed);
		createTrackbar("Lower g", "mask", &lower_g, 255, on_hue_changed);
		//createTrackbar("Upper g", "mask", &upper_g, 255, on_hue_changed);
		createTrackbar("Lower b", "mask", &lower_b, 255, on_hue_changed);
		//createTrackbar("Upper b", "mask", &upper_b, 255, on_hue_changed);
		on_hue_changed(0, 0);
		imshow("frame", frame);
		imshow("mask", mask);    
		if (waitKey(10) == 27)
			break;
	}
	return 0;
}

void on_hue_changed(int, void*) {
	
	//cvtColor(frame, cvt_frame, COLOR_BGR2GRAY);
	Mat k;
	//Mat edge;
	//createTrackbar("Threshold", "cvt_frame", 0, 255, on_threshold);	
	//Scalar upperb(upper_r, upper_g, upper_b);
	Scalar lowerb(lower_r, lower_g, lower_b);
	//Scalar lowerb;
	Scalar upperb(255,255,255);
	Point centroid;
	inRange(frame, lowerb, upperb, mask);
	mask = ~mask;
	//threshold(cvt_frame, mask, lower_r, 255,THRESH_BINARY);
	erode(mask, mask, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);
	//erode(mask, mask, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);

	/*Canny(mask, edge, 50, 200);
	vector<Vec4i> lines;
	HoughLinesP(edge, lines, 1, CV_PI / 180, 160, 50, 5);
	for (Vec4i l : lines) {
		line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);

	}*/




	vector<vector<Point>> contours;  //외곽선 검출
	vector<vector<Point>> points;  // 점들
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//cerr << contours.size() << endl;
	//drawContours(frame, contours,-1, LINE_8);
	for (vector<Point>& pts : contours) {
		if (contourArea(pts) < 400)
			continue;
		vector<Point>  approx;
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);
		int vtc = (int)approx.size();

		if (vtc == 4) {
			//cerr << approx.size() << endl;
			//double size = contourArea(approx);
			Label(frame, pts, "window detect");
	}
	}
	//for (int i = 0; i < contours.size(); i++) {
		//int size = contours[i].size();
		//centroid = contours[i][0] - contours[i][(3 * size) / 4];
		//circle(frame, centroid, 30, Scalar(255, 255, 0), 2);
	//}*/
	

}

void Label(Mat& img, const vector<Point>& pts, const String& label)
{
	Rect rc = boundingRect(pts);
	rectangle(img, rc, Scalar(255, 0, 0), 1);
	Point x = rc.tl();
	Point y = rc.br();
	Point centroid = (x + y) / 2;
	circle(img, centroid, 5, Scalar(255, 255, 255), 2);
	putText(img, label, rc.tl(), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
	cerr << "bouding box area:"<< rc.area() << endl;
	vector<int> params;
	params.push_back(IMWRITE_JPEG_QUALITY);
	params.push_back(95);
	Mat object = img(rc);
	imwrite("object.jpg", object, params);
}



	


//cv::Mat src, dst;
//cv::Mat brightness;

/*void draw_bounding_box();
int main(void) {
	draw_bounding_box();
}
void draw_bounding_box() {
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "camera open failed!" << endl;
		return;
	}

	Mat frame;
	Mat gray_scale_frame;

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, gray_scale_frame, COLOR_BGR2GRAY);
		Mat bin;
		threshold(gray_scale_frame, bin, 120, 255, THRESH_BINARY_INV);
		Mat labels, stats, centroids;
		int cnt = connectedComponentsWithStats(bin, labels, stats, centroids);
		Mat RGB_frame;
		cvtColor(gray_scale_frame, RGB_frame, COLOR_GRAY2BGR);
		for (int i = 1; i < cnt; i++) {
			int* p = stats.ptr<int>(i);
			if (p[4] < 20) continue;
			rectangle(RGB_frame, Rect(p[0], p[1], p[2], p[3]), Scalar(255, 0, 0), 2);
		}
		imshow("frame", frame);
		imshow("RGB_frame", RGB_frame);
		if (waitKey(10) == 27)
			break;
	}
}*/


/*void detect_rect(); //도형 탐지로 접근
void Label(Mat& img, const vector<Point>& pts, const String& label);
int main(void)
{
	detect_rect();
}

void detect_rect() {
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "camera open failed!" << endl;
		return;
	}

	Mat frame;
	Mat gray_scale_frame;

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, gray_scale_frame, COLOR_BGR2GRAY);
		Mat binary_video;
		threshold(gray_scale_frame, binary_video, 100, 255, THRESH_TOZERO);
		vector<vector<Point>> contours;
		findContours(binary_video, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		for (vector<Point>& pts : contours) {
			if (contourArea(pts) < 200)
				continue;
			vector<Point>  approx;
			approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);
			int vtc = (int)approx.size();

			if (vtc == 4) {
				Label(frame, pts, "window is detected");
			}
		}
		imshow("binary_video", binary_video);
		imshow("frame", frame);
		if (waitKey(10) == 27)
			break;

	}
}
void Label(Mat& img, const vector<Point>& pts, const String& label) 
{
   Rect rc = boundingRect(pts);
   rectangle(img, rc, Scalar(255, 0, 0), 1);
   putText(img, label, rc.tl(), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
}*/


/*void camera_in(); //엣지 검출

int main(void) {
	camera_in();
}

void camera_in() {

	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "camera open failed!" << endl;
		return;
	}

	Mat frame;
	Mat gray_scale_frame;
	Mat edge;
	

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, gray_scale_frame, COLOR_BGR2GRAY);
		Canny(gray_scale_frame, edge, 50, 100);
		//cvtColor(edge, edge_frame, COLOR_GRAY2BGR);

		vector<Vec4i> lines;
		HoughLinesP(edge, lines, 1, CV_PI / 180, 10, 50, 5);
		Mat edge_frame;
		cvtColor(edge, edge_frame, COLOR_GRAY2BGR);
		for (Vec4i l : lines) {
			line(edge_frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);

		}
		/*HoughLines(edge, lines, 1, CV_PI / 180, 250);
		for (size_t i = 0; i < lines.size(); i++) {
			float r = lines[i][0], t = lines[i][1];
			double cos_t = cos(t), sin_t = sin(t);
			double x0 = r * cos_t, y0 = r * sin_t;
			double alpha = 1000;
			Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
			Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
			line(edge_frame, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
		}*/

		/*imshow("edge_frame", edge_frame);
		//imshow("frame", frame);
		if (waitKey(10) == 27)
			break;

	}
	destroyAllWindows();
}*/



/*void Label(Mat& img, const vector<Point>& pts, const String& label, Point centroid);  //gray_scale변환 후 사각형 검출 알고리즘
void on_threshold(int pos, void* userdata);
int main(void) {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "카메라 열기 실패" << endl;
		return 0;
	}

	Mat frame;
	Mat gray_scale_frame;
	Point centroid;


	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, gray_scale_frame, COLOR_BGR2GRAY);
		namedWindow("bin");
		createTrackbar("Threshold", "bin", 0, 255, on_threshold);
		
		//setTrackbarPos("Threshold", "bin", 128);
		//threshold(gray_scale_frame, bin, 128, 255, THRESH_BINARY);
		//cv::threshold(gray_scale_frame, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

		vector<vector<Point>> contours;
		findContours(gray_scale_frame, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		//Mat BGR_frame;
		//cvtColor(bin, BGR_frame, COLOR_GRAY2BGR);
		Scalar c(255, 255, 0);
		//drawContours(frame, contours,-1, c, 2);
		cerr << contours.size() << endl;
		for (vector<Point>& pts : contours) {
			if (contourArea(pts) < 400)
				continue;
			vector<Point>  approx;
			approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);
			int vtc = (int)approx.size();

			if (vtc == 4) {
				cerr << pts.size() << endl;
				int size = pts.size();
				centroid = pts[0] - pts[(size * 3) / 4];
				Label(frame, pts, "window detect", centroid);
			}
		}
		//for (int i = 0; i < contours.size(); i++) {
		  // int size = contours[i].size();
		   //centroid = contours[i][0] - contours[i][(3 * size) / 4];
		   //circle(frame, centroid, 30, Scalar(255, 255, 0), 2);
		//}

		imshow("frame", frame);
		if (waitKey(10) == 27)
			break;

	}
	return 0;
}

void on_threshold(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;
	Mat dst;
	threshold(src, dst, pos, 255, THRESH_BINARY);
	imshow("dst", dst);
}



void Label(Mat& img, const vector<Point>& pts, const String& label, Point centroid)
{
	Rect rc = boundingRect(pts);
	rectangle(img, rc, Scalar(255, 0, 0), 1);
	circle(img, centroid, 20, Scalar(255, 255, 255), 2);
	putText(img, label, rc.tl(), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
}*/


/*
int main(void)
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "카메라 열기 실패" << endl;
		return 0;
	}

	Mat frame;
	Mat gray_scale_frame;
	Point centroid;


	while (true) {
		cap >> frame;

		// convert to grayscale (you could load as grayscale instead)
		cv::Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// compute mask (you could use a simple threshold if the image is always as good as the one you provided)
		cv::Mat mask;
		cv::threshold(gray, mask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

		// find contours (if always so easy to segment as your image, you could just add the black/rect pixels to a vector)
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		/// Draw contours and find biggest contour (if there are other contours in the image, we assume the biggest one is the desired rect)
		// drawing here is only for demonstration!
		int biggestContourIdx = -1;
		float biggestContourArea = 0;
		cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			cv::Scalar color = cv::Scalar(0, 100, 0);
			drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());

			float ctArea = cv::contourArea(contours[i]);
			if (ctArea > biggestContourArea)
			{
				biggestContourArea = ctArea;
				biggestContourIdx = i;
			}
		}

		// if no contour found
		if (biggestContourIdx < 0)
		{
			std::cout << "no contour found" << std::endl;
			return 1;
		}

		// compute the rotated bounding rect of the biggest contour! (this is the part that does what you want/need)
		cv::RotatedRect boundingBox = cv::minAreaRect(contours[biggestContourIdx]);
		// one thing to remark: this will compute the OUTER boundary box, so maybe you have to erode/dilate if you want something between the ragged lines

		// draw the rotated rect
		cv::Point2f corners[4];
		boundingBox.points(corners);
		cv::line(drawing, corners[0], corners[1], cv::Scalar(255, 255, 255));
		cv::line(drawing, corners[1], corners[2], cv::Scalar(255, 255, 255));
		cv::line(drawing, corners[2], corners[3], cv::Scalar(255, 255, 255));
		cv::line(drawing, corners[3], corners[0], cv::Scalar(255, 255, 255));

		// display
		cv::imshow("frame", frame);
		cv::imshow("drawing", drawing);
		//cv::waitKey(0);

		//cv::imwrite("rotatedRect.png", drawing);


		if (frame.empty())
			break;
		cvtColor(frame, gray_scale_frame, COLOR_BGR2GRAY);
		if (waitKey(10) == 27)
			break;

	}

	return 0;
}
*/

