#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/highgui/highgui_c.h>   // C
#include <opencv2/imgproc/imgproc_c.h>   // C
#include "yolo_v2_class.hpp" 

using namespace std;
using namespace cv;

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
	if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
	std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
	    << ", w = " << i.w << ", h = " << i.h
	    << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
	int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
	cv::Scalar color = obj_id_to_color(i.obj_id);
	cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
	if (obj_names.size() > i.obj_id) {
	    std::string obj_name = obj_names[i.obj_id];
	    if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
	    cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
	    int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
	    max_width = std::max(max_width, (int)i.w + 2);
	    //max_width = std::max(max_width, 283);
	    std::string coords_3d;
	    if (!std::isnan(i.z_3d)) {
		std::stringstream ss;
		ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
		coords_3d = ss.str();
		cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
		int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
		if (max_width_3d > max_width) max_width = max_width_3d;
	    }

	    cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
		    cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
		    color, CV_FILLED, 8, 0);
	    putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
	    if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
	}
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
	std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
	putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

void detect_video(string video_path) {
    std::string  names_file = "/home/xzw/dataset/dazu/myData/myData.names";
    std::string  cfg_file = "/home/xzw/WorkSpace/darknet/cfg/my_yolov3.cfg";
    std::string  weights_file = "/home/xzw/dataset/dazu/myData/backup/my_yolov3.backup";

    float const thresh = 0.25;

    Detector detector(cfg_file, weights_file);

    auto obj_names = objects_names_from_file(names_file);
    VideoCapture vc(video_path);
    cv::Mat mat_img;
    while (vc.read(mat_img)) {
        auto start = std::chrono::steady_clock::now();
        std::vector<bbox_t> result_vec = detector.detect(mat_img);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> spent = end - start;
        std::cout << " Time: " << spent.count() << " sec \n";

        //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
        draw_boxes(mat_img, result_vec, obj_names);
        cv::imshow("window name", mat_img);
        show_console_result(result_vec, obj_names);
        cv::waitKey(0);
    }
    
}

int main(int argc, char ** argv)
{
    detect_video(argv[1]);
    // std::string  names_file = "/home/xzw/WorkSpace/darknet-so/data/coco.names";
    // std::string  cfg_file = "/home/xzw/WorkSpace/darknet-so/cfg/yolov3-spp.cfg";
    // std::string  weights_file = "/home/xzw/WorkSpace/weights/yolov3-spp.weights";
    // std::string filename;

    // if (argc > 4) {    //voc.names yolo-voc.cfg yolo-voc.weights test.mp4
	// names_file = argv[1];
	// cfg_file = argv[2];
	// weights_file = argv[3];
	// filename = argv[4];
    // }
    // else if (argc > 1) 
    // {
	// filename = argv[1];
    // }

    // float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.2;

    // Detector detector(cfg_file, weights_file);

    // auto obj_names = objects_names_from_file(names_file);

    // cv::Mat mat_img = cv::imread(filename);

    // auto start = std::chrono::steady_clock::now();
    // std::vector<bbox_t> result_vec = detector.detect(mat_img);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> spent = end - start;
    // std::cout << " Time: " << spent.count() << " sec \n";

    // //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
    // draw_boxes(mat_img, result_vec, obj_names);
    // cv::imshow("window name", mat_img);
    // show_console_result(result_vec, obj_names);
    // cv::waitKey(0);
    return 0;
}