#include <opencv2/core.hpp>
using namespace cv;
double LineLen(cv::Point line_a, cv::Point line_b)
{
	return sqrt(pow(line_a.x - line_b.x, 2) + pow(line_a.y - line_b.y, 2));
}

// a signed distance
double Dist_Pt_To_Line(cv::Point2f line_a, cv::Point2f line_b, cv::Point2f d)
{
	if (fabs(line_b.x - line_a.x) > 1e-10)
	{
		double a = (line_b.y - line_a.y) / (line_b.x - line_a.x);
		double b = -1;
		double c = line_b.y - a*line_b.x;

		double base = sqrt(pow(a, 2) + pow(b, 2));
		double upper_term = fabs(a*d.x + b*d.y + c);
		double dist = upper_term / base;

		double proj_x = (b*(b*d.x - a*d.y) - a*c) / (pow(base, 2));
		return -1 * dist*(proj_x >= d.x ? 1 : -1);
	}
	else
	{
		return d.x - line_b.x;
	}

	//double y_delta = line_b.y - line_a.y;
	//double x_delta = line_b.x - line_a.x;
	//double base = sqrt(pow(y_delta, 2) + pow(x_delta, 2));
	//double upper_term = y_delta*c.x-x_delta*c.y+line_b.x*line_a.y-line_b.y*line_a.x;
	//return upper_term / base;
}
//
//double quad_wrap(double a)
//{
//	if (a < 0) a += PI;
//	int quadrant = floor(a / (PI*0.5f));
//	switch (quadrant)
//	{
//	case 0:
//		return a;
//		break;
//	case 1:
//		return PI - a;
//		break;
//	case 2:
//		return a - PI;
//		break;
//	case 3:
//		return 2 * PI - a;
//		break;
//	default:
//		return a;
//		break;
//	}
//}
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

void Line_To_Polar_Coord(cv::Point2f a, cv::Point2f b, double &rho, double &theta)
{
	//Calcaluate the 'm' and 'c' in y=mx+c
	if (fabs(b.x - a.x) > 1e-10)
	{
		double m = (b.y - a.y) / (b.x - a.x);
		double c = b.y - m*b.x;
		theta = atan(-1.0f / m);
		rho = c*sin(theta);
	}
	else
	{
		theta = 0;
		rho = b.x;
	}

}