#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "boost/filesystem.hpp"
#include <boost/algorithm/string.hpp>  
#include <opencv2/video/tracking.hpp>

#define PI 3.1415926
#define PATH_TO_IMAGES "F:/lane_marking_detection/calibration_image/basler/video/video1"
#define Calibration_File "F:/lane_marking_detection/calibration_image/result_copy.yml"
#define Need_Rotate_R 1
#define RATIO 0.5
#define LANE_PHYSICAL_WIDTH		(303)
#define LANE_PHYSICAL_WIDTH_THD		(15)
#define PARALLEL_LINE_ANGLE_DIFF	((2*PI)/180)
#define SKIP_FRAME_RATIO	(2)
#define MAX_MISSING_LINE_FRAME	(4)

using namespace cv;

void cvShowManyImages(char* title, int nArgs, ...);

struct PairOfLine
{
	int x;
	int y;
	double score;
	double anglediff;
	double width;
};

double quad_wrap(double a, int quad=-1)
{
	int quadrant = floor(a / (PI*0.5f));
	if (quad >= 0) quadrant = quad;
	switch (quadrant)
	{
		case 0:
			return a;
			break;
		case 1:
			return PI - a;
			break;
		default:
			return a;
			break;
	}
}
//
//double quad_unwrap(double a, int quad)
//{
//	switch (quad)
//	{
//	case 0:
//		return a;
//		break;
//	case 1:
//		return PI - a;
//		break;
//	case 2:
//		return a + PI;
//		break;
//	case 3:
//		return 2 * PI - a;
//		break;
//	default:
//		return a;
//		break;
//	}
//}

std::vector<std::string> GetFileNamesInDirectory(const std::string directory)
{
	std::vector<std::string> files;
	try {
		boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
		for (boost::filesystem::directory_iterator i(directory); i != end_itr; ++i) {
			// Skip if not a file
			if (!boost::filesystem::is_regular_file(i->status()))
				continue;

			// File matches, store it
			std::string tmp = i->path().extension().string();
			boost::algorithm::to_lower(tmp);
			if (tmp == ".bmp")	files.push_back(i->path().string());
		}
	}
	catch (std::exception& e) { std::cerr << e.what() << std::endl; }
	return files;
}

int main2(int argc, char* argv[])
{
	//Read calibration yml
	cv::FileStorage fs(Calibration_File, FileStorage::READ);
	cv::Mat Mint, Mext, distCoeffs;
	fs["Mint"] >> Mint;
	std::cout << "Mint: " << Mint << std::endl;
	fs["Mext"] >> Mext;
	if (Need_Rotate_R) Mext.colRange(0, 3) = Mext.colRange(0, 3).t();
	std::cout << "Mext: " << Mext << std::endl;
	fs["distCoeffs"] >> distCoeffs;
	std::cout << "distCoeffs: " << distCoeffs << std::endl;
	fs.release();


	// Get all *.jpg in the GT luggage database and testcase database
	std::vector<std::string> vFileNames = GetFileNamesInDirectory(PATH_TO_IMAGES);
	std::sort(vFileNames.begin(), vFileNames.end());
	auto nImage = vFileNames.size();

	//Do the transformation matrix
	cv::Mat temp, tempgray, tempimage = imread(vFileNames[0]);
	cvtColor(tempimage, tempgray, CV_RGB2GRAY);
	undistort(tempgray, temp, Mint, distCoeffs);

	double diagonal = sqrt(temp.rows*temp.rows + temp.cols*temp.cols);
	std::cout << "diagonal: " << diagonal << std::endl;

	cv::Mat obj_corners = cv::Mat::ones(3, 5, Mint.type());
	cv::Mat scene_corners = cv::Mat::zeros(3, 5, Mint.type());
	obj_corners.at<double>(0, 0) = 0;
	obj_corners.at<double>(1, 0) = 0;
	obj_corners.at<double>(0, 1) = temp.cols;
	obj_corners.at<double>(1, 1) = 0;
	obj_corners.at<double>(0, 2) = 0;
	obj_corners.at<double>(1, 2) = temp.rows;
	obj_corners.at<double>(0, 3) = temp.cols;
	obj_corners.at<double>(1, 3) = temp.rows;
	obj_corners.at<double>(0, 4) = temp.cols*0.5f;
	obj_corners.at<double>(1, 4) = temp.rows*0.5f;

	cv::Mat persT = Mext.colRange(0, 2);
	cv::hconcat(persT, Mext.col(3), persT);
	persT = (Mint*persT).inv();
	scene_corners = persT*obj_corners;

	double minx = 1e10, miny = 1e10, maxx = 0, maxy = 0;
	for (int i = 0; i < scene_corners.cols; i++)
	{
		scene_corners.at<double>(0, i) /= scene_corners.at<double>(2, i);
		scene_corners.at<double>(1, i) /= scene_corners.at<double>(2, i);
		if (scene_corners.at<double>(0, i) < minx) minx = scene_corners.at<double>(0, i);
		if (scene_corners.at<double>(0, i) > maxx) maxx = scene_corners.at<double>(0, i);
		if (scene_corners.at<double>(1, i) < miny) miny = scene_corners.at<double>(1, i);
		if (scene_corners.at<double>(1, i) > maxy) maxy = scene_corners.at<double>(1, i);
	}

	double xsize = maxx - minx;
	double ysize = maxy - miny;
	double ratio = (RATIO* diagonal) / min(xsize, ysize);
	std::cout << "ratio: " << ratio << std::endl;
	double xoffset = minx < 0 ? -1 * ratio*minx : 0;
	double yoffset = miny < 0 ? -1 * ratio*miny : 0;
	double xcenter = scene_corners.at<double>(0, 4)*ratio + xoffset;
	double ycenter = scene_corners.at<double>(1, 4)*ratio + yoffset;

	cv::Mat correctionmat = cv::Mat::eye(3, 3, Mint.type());
	correctionmat.at<double>(0, 0) = ratio;
	correctionmat.at<double>(1, 1) = ratio;
	correctionmat.at<double>(0, 2) = xoffset;
	correctionmat.at<double>(1, 2) = yoffset;

	persT = correctionmat*persT;

	//Setup required matrix for kalman filter
	int stateSize = 10;
	int measSize = 5;
	int contrSize = 0;

	unsigned int type = CV_64F;
	cv::KalmanFilter kf(stateSize, measSize, contrSize, type);   
	cv::setIdentity(kf.transitionMatrix);
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<double>(0) = 1.0f;
	kf.measurementMatrix.at<double>(12) = 1.0f;
	kf.measurementMatrix.at<double>(24) = 1.0f;
	kf.measurementMatrix.at<double>(36) = 1.0f;
	kf.measurementMatrix.at<double>(48) = 1.0f;

	cv::setIdentity(kf.processNoiseCov, Scalar::all(1e-1));
	cv::setIdentity(kf.measurementNoiseCov, Scalar::all(1e-3));
	cv::setIdentity(kf.errorCovPost, Scalar::all(1e-2));
	Mat_<double> measurement(measSize, 1);

	std::cout << "Number of images: " << nImage << std::endl;
	std::cout << "Wrapped image size: " << floor(xsize*ratio) + 1 << ", " << floor(ysize*ratio) + 1 << std::endl;
	std::cout << "Start lane detection...\n" << std::endl;
	int64	stime, etime, stime2, etime2;
	double ticks = 0;
	bool filterFirsttime = true;
	int sign1=1, sign2=1, quad1=0, quad2=0;
	int misslinecounter = 0;
	int mx = 0, my = 0;

	for (int i = 0; i < nImage; i+= SKIP_FRAME_RATIO)
	{
		stime = cv::getTickCount();

		stime2 = cv::getTickCount();
		cv::Mat gray, image = imread(vFileNames[i]);
		cvtColor(image, gray, CV_RGB2GRAY);
		etime2 = cv::getTickCount();
		printf("Read and to gray %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		//undistort
		stime2 = cv::getTickCount();
		cv::Mat undistored, displayresize;
		undistort(gray, undistored, Mint, distCoeffs);
		etime2 = cv::getTickCount();
		printf("Undistort %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());
		cv::resize(undistored, displayresize, cv::Size(0, 0), 0.7f, 0.7f);

		// Display Canny image
		//if (1) {
		//	imshow("Original view", displayresize);
		//}

		//Wrapped
		stime2 = cv::getTickCount();
		cv::Mat wrapped;
		warpPerspective(undistored, wrapped, persT, cv::Size(floor(xsize*ratio) + 1, floor(ysize*ratio) + 1));

		etime2 = cv::getTickCount();
		printf("wrap %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		// Canny algorithm
		stime2 = cv::getTickCount();
		Mat blurred, contours, mask, maskerode;
		blur(wrapped, blurred, cv::Size(5, 5));
		double cannythd = 25;
		Canny(blurred, contours, cannythd, 2 * cannythd);
		threshold(wrapped, mask, 1, 255, THRESH_BINARY);
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * 2 + 1, 2 * 2 + 1),
			Point(2, 2));
		erode(mask, maskerode, element);
		contours = contours.mul(maskerode);
		etime2 = cv::getTickCount();
		printf("Filter and canny %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		// Display Canny image
		//if (1) {
		//	imshow("Canny", contours);
		//}

		stime2 = cv::getTickCount();
		std::vector<Vec2f> lines;
		HoughLines(contours, lines, 1, PI / 180, 200* RATIO);

		cv::Mat houghresult = wrapped.clone();

		// Draw the limes
		int count = 0;
		std::vector<Vec2f>::const_iterator it = lines.begin();
		while (it != lines.end()) {
			float rho = (*it)[0];   // first element is distance rho
			float theta = (*it)[1]; // second element is angle theta

			Point pt1(rho / cos(theta), 0);
			Point pt2((rho - houghresult.rows*sin(theta)) / cos(theta), houghresult.rows);
			line(houghresult, pt1, pt2, Scalar(255), 2);

			std::cout << "line" << count << ": (" << rho << "," << theta << ")\n"; 
			++it; count++;
		}

		//Find pair of parallel lines
		std::vector<PairOfLine> pairs;
		for (int j = 0; j < lines.size() - 1; j++)
		{
			for (int k = j+1; k < lines.size(); k++)
			{
				double anglediff = fabs(lines[j][1] - lines[k][1]);
				double anglescore = anglediff / PARALLEL_LINE_ANGLE_DIFF;
				double width = fabs(lines[j][0] - lines[k][0]);
				double widthscore = fabs(width/ratio - LANE_PHYSICAL_WIDTH) / LANE_PHYSICAL_WIDTH_THD;

				double centrerhoshiftscore = 1.0f;

				if (anglescore < 1 && widthscore < 1)
				{
					if (!filterFirsttime)
					{
						double prevcentrerho = (kf.statePre.at<double>(0) + kf.statePre.at<double>(4))*0.5f;
						double currcenterrho = (fabs(lines[j][0]) + fabs(lines[k][0]))*0.5f;
						centrerhoshiftscore = fabs(prevcentrerho - currcenterrho) / 40;
						printf("%d, %d: prev: %f, curr: %f, anglescore: %f, widthscore: %f, centrerhoshiftscore: %f\n",
							j, k, prevcentrerho, currcenterrho, anglescore, widthscore, centrerhoshiftscore);
					}

					PairOfLine temp;
					temp.x = j; temp.y = k;
					temp.score = 1 - anglescore - widthscore - centrerhoshiftscore;
					temp.anglediff = anglediff;
					temp.width = width;
					pairs.emplace_back(temp);
				}
			}
		}
		printf("There are %d candidate for parallel line pairs\n", pairs.size());

		//Sort them by their score
		int maxID = 0;
		double maxscore = -99999.9f;
		cv::Mat parallelresult = wrapped.clone();
		if (pairs.size() > 0)
		{
			misslinecounter=0;
			for (int j = 0; j < pairs.size(); j++)
			{
				if (pairs[j].score > maxscore) {
					maxscore = pairs[j].score;
					maxID = j;
					mx = pairs[j].x; my = pairs[j].y;
					// Sort so that mx corresponding to the line with smaller rho
					if (lines[mx][0] > lines[my][0]) {
						int temp = mx;
						mx = my; my = temp;
					}
				}
				printf("Candidate %d,x: %d, y: %d, score: %f, anglediff: %f, width: %f\n", j, pairs[j].x, pairs[j].y, pairs[j].score,
					pairs[j].anglediff, pairs[j].width);
			}
			Point pt11(lines[mx][0] / cos(lines[mx][1]), 0);
			Point pt12((lines[mx][0] - parallelresult.rows*sin(lines[mx][1])) / cos(lines[mx][1]), parallelresult.rows);
			line(parallelresult, pt11, pt12, Scalar(255), 2);
			Point pt21(lines[my][0] / cos(lines[my][1]), 0);
			Point pt22((lines[my][0] - parallelresult.rows*sin(lines[my][1])) / cos(lines[my][1]), parallelresult.rows);
			line(parallelresult, pt21, pt22, Scalar(255), 2);
		}
		else
		{
			misslinecounter++;
		}


		//if (1) {
		//	imshow("Parallel pair", parallelresult);
		//}

		etime2 = cv::getTickCount();
		printf("Hough %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());

		// Display the detected line image
		//if (1) {
		//	imshow("After hough", houghresult);
		//}

		etime = cv::getTickCount();
		double timeused = (double)(etime - stime) / (double)cv::getTickFrequency();

		//Kalman filter to track the parallel line
		double precTick = ticks;
		ticks = (double)cv::getTickCount();
		double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
		std::cout << "dT: " << dT << std::endl;
		cv::Mat filterresult;

		if (filterFirsttime == true)
		{
			if (pairs.size() > 0 && dT < 0.5)
			{
				kf.statePre.at<double>(0) = fabs(lines[mx][0]);
				kf.statePre.at<double>(1) = 0.0f;
				kf.statePre.at<double>(2) = quad_wrap(lines[mx][1]);
				kf.statePre.at<double>(3) = 0.0f;
				kf.statePre.at<double>(4) = fabs(lines[my][0]);
				kf.statePre.at<double>(5) = 0.0f;
				kf.statePre.at<double>(6) = quad_wrap(lines[my][1]);
				kf.statePre.at<double>(7) = 0.0f;
				kf.statePre.at<double>(8) = pairs[maxID].width;
				kf.statePre.at<double>(9) = 0.0f;

				kf.statePost.at<double>(0) = fabs(lines[mx][0]);
				kf.statePost.at<double>(1) = 0.0f;
				kf.statePost.at<double>(2) = quad_wrap(lines[mx][1]);
				kf.statePost.at<double>(3) = 0.0f;
				kf.statePost.at<double>(4) = fabs(lines[my][0]);
				kf.statePost.at<double>(5) = 0.0f;
				kf.statePost.at<double>(6) = quad_wrap(lines[my][1]);
				kf.statePost.at<double>(7) = 0.0f;
				kf.statePost.at<double>(8) = pairs[maxID].width;
				kf.statePost.at<double>(9) = 0.0f;
				filterFirsttime = false;
				std::cout << "Kalman initialized at: " << kf.statePre.t() << std::endl;
			}

		}
		else
		{
			filterresult = wrapped.clone();
			stime2 = cv::getTickCount();

			kf.transitionMatrix.at<double>(1) = dT;
			kf.transitionMatrix.at<double>(23) = dT;
			kf.transitionMatrix.at<double>(45) = dT;
			kf.transitionMatrix.at<double>(67) = dT;
			kf.transitionMatrix.at<double>(89) = dT;

			cv::Mat prediction = kf.predict();
			std::cout << "prediction: " << prediction.t() << std::endl;

			//Update measurement
			cv::Mat estimated;
			if (pairs.size() > 0)
			{
				measurement.at<double>(0) = fabs(lines[mx][0]);
				measurement.at<double>(1) = quad_wrap(lines[mx][1]);
				measurement.at<double>(2) = fabs(lines[my][0]);
				measurement.at<double>(3) = quad_wrap(lines[my][1]);
				measurement.at<double>(4) = pairs[maxID].width;
				estimated = kf.correct(measurement);
				std::cout << "estimated: " << estimated.t() << std::endl;
				printf("Measurement: %f, %f, %f, %f, %f\n", lines[mx][0], lines[mx][1], lines[my][0], lines[my][1], pairs[maxID].width);
				sign1 = lines[mx][0] < 0 ? -1 : 1;
				sign2 = lines[my][0] < 0 ? -1 : 1;
				quad1 = floor(lines[mx][1] / (PI*0.5f));
				quad2 = floor(lines[my][1] / (PI*0.5f));
			}
			else
			{
				estimated = prediction.clone();
				//sign1 and sign2 keep using last round result
			}

			if (misslinecounter < MAX_MISSING_LINE_FRAME)
			{
				Point pt11(sign1*estimated.at<double>(0) / cos(quad_wrap(estimated.at<double>(2),quad1)), 0);
				Point pt12((sign1*estimated.at<double>(0) - filterresult.rows*sin(quad_wrap(estimated.at<double>(2),
					quad1))) / cos(quad_wrap(estimated.at<double>(2), quad1)), filterresult.rows);
				line(filterresult, pt11, pt12, Scalar(255), 2);

				Point pt21(sign2*estimated.at<double>(4) / cos(quad_wrap(estimated.at<double>(6), quad2)), 0);
				Point pt22((sign2*estimated.at<double>(4) - filterresult.rows*sin(quad_wrap(estimated.at<double>(6),
					quad2))) / cos(quad_wrap(estimated.at<double>(6), quad2)), filterresult.rows);
				line(filterresult, pt21, pt22, Scalar(255), 2);
			}


			//if (1) {
			//	imshow("Kalman filter", filterresult);
			//}
		}


		etime2 = cv::getTickCount();
		printf("Kalman filter %f seconds\n", (double)(etime2 - stime2) / (double)cv::getTickFrequency());


		std::stringstream stream;
		stream << "Lines Segments: " << lines.size() << "   FPS: " << 1.0 / timeused;

		cv::Mat FPSimg = wrapped.clone();
		putText(FPSimg, stream.str(), Point(40, 70), 2, 0.6, Scalar(0, 0, 255), 0);
		//Draw the center of projected scene coord
		line(filterresult, cv::Point(xcenter - 20, ycenter), cv::Point( xcenter + 20, ycenter), Scalar(255), 2);
		line(filterresult, cv::Point(xcenter, ycenter - 20), cv::Point(xcenter, ycenter + 20), Scalar(255), 2);
		//imshow("wrapped", FPSimg);
		cvShowManyImages("Result image", 6,displayresize, contours, houghresult, parallelresult, filterresult, FPSimg);

		//char key = (char)waitKey(5);
		//if (key == 'q')	break; 
		lines.clear();
	}
}



