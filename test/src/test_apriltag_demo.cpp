/**
 * BSD 3-Clause License
 *
 * This file is part of the Basalt project.
 * https://gitlab.com/VladyslavUsenko/basalt.git
 *
 * Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
 * All rights reserved.
 *
 * AprilTag标定板识别Demo程序
 * 读取文件夹下的图片，检测apriltag并显示结果
 */

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <basalt/utils/apriltag.h>
#include <basalt/image/image.h>

using namespace std;
namespace fs = std::filesystem;

/**
 * @brief 打印使用帮助
 */
void printUsage(const char* progName) {
    cout << "AprilTag标定板识别程序\n"
         << "用法: " << progName << " <image_folder> [options]\n\n"
         << "参数:\n"
         << "  image_folder    包含标定图像的文件夹路径\n\n"
         << "选项:\n"
         << "  -r <rows>       标签行数（默认：6）\n"
         << "  -c <cols>       标签列数（默认：6）\n"
         << "  -s <size>       标签尺寸[m]（默认：0.088）\n"
         << "  -p <spacing>    标签间距比例（默认：0.3）\n"
         << "  -b <border>     黑边宽度（默认：2）\n"
         << "  -v              显示检测结果视频\n"
         << "  -o <output>     输出文件路径（默认：detections.txt）\n"
         << "  -h              显示此帮助信息\n"
         << endl;
}

/**
 * @brief 解析命令行参数
 */
bool parseArguments(int argc, char* argv[], 
                    string& imageFolder,
                    int& tagRows, int& tagCols,
                    double& tagSize, double& tagSpacing,
                    int& blackBorder,
                    bool& showVideo,
                    string& outputFile) {
    if (argc < 2) {
        printUsage(argv[0]);
        return false;
    }

    // 默认值
    imageFolder = "";
    tagRows = 6;
    tagCols = 6;
    tagSize = 0.088;
    tagSpacing = 0.3;
    blackBorder = 2;
    showVideo = false;
    outputFile = "detections.txt";

    // 解析参数
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "-r" && i + 1 < argc) {
            tagRows = stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            tagCols = stoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            tagSize = stod(argv[++i]);
        } else if (arg == "-p" && i + 1 < argc) {
            tagSpacing = stod(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {
            blackBorder = stoi(argv[++i]);
        } else if (arg == "-v") {
            showVideo = true;
        } else if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (imageFolder.empty()) {
            imageFolder = arg;
        } else {
            cerr << "未知参数: " << arg << endl;
            printUsage(argv[0]);
            return false;
        }
    }

    if (imageFolder.empty()) {
        cerr << "错误: 必须指定图像文件夹路径" << endl;
        printUsage(argv[0]);
        return false;
    }

    return true;
}

/**
 * @brief 获取文件夹下所有图片文件
 */
vector<string> getImageFiles(const string& folder) {
    vector<string> imageFiles;
    
    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        cerr << "错误: 文件夹不存在或不是有效目录: " << folder << endl;
        return imageFiles;
    }

    vector<string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"};
    
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            string ext = entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    // 按文件名排序
    sort(imageFiles.begin(), imageFiles.end());
    
    return imageFiles;
}

/**
 * @brief 将OpenCV Mat转换为ManagedImage
 */
basalt::ManagedImage<uint16_t> cvMatToManagedImage(const cv::Mat& img) {
    basalt::ManagedImage<uint16_t> managedImg(img.cols, img.rows);
    
    if (img.type() == CV_8UC1) {
        // 灰度图
        const uint8_t* src = img.ptr<uint8_t>();
        uint16_t* dst = managedImg.ptr;
        for (size_t i = 0; i < img.total(); i++) {
            dst[i] = static_cast<uint16_t>(src[i]) << 8;
        }
    } else if (img.type() == CV_8UC3) {
        // BGR彩色图转灰度
        const uint8_t* src = img.ptr<uint8_t>();
        uint16_t* dst = managedImg.ptr;
        for (size_t i = 0; i < img.total(); i++) {
            // 使用BGR转灰度的标准权重
            uint8_t gray = static_cast<uint8_t>(
                0.299 * src[i * 3 + 2] +  // R
                0.587 * src[i * 3 + 1] +  // G
                0.114 * src[i * 3 + 0]     // B
            );
            dst[i] = static_cast<uint16_t>(gray) << 8;
        }
    } else if (img.type() == CV_16UC1) {
        // 16位灰度图
        memcpy(managedImg.ptr, img.ptr<uint16_t>(), img.total() * sizeof(uint16_t));
    } else {
        cerr << "警告: 不支持的图像格式，尝试转换为灰度图" << endl;
        cv::Mat gray;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img.clone();
        }
        return cvMatToManagedImage(gray);
    }
    
    return managedImg;
}

/**
 * @brief 在图像上绘制检测到的角点
 */
void drawCorners(cv::Mat& img, 
                 const Eigen::aligned_vector<Eigen::Vector2d>& corners,
                 const std::vector<int>& ids,
                 const std::vector<double>& radii,
                 const cv::Scalar& color) {
    for (size_t i = 0; i < corners.size(); i++) {
        cv::Point2d pt(corners[i].x(), corners[i].y());
        // double radius = radii[i];
        
        // 绘制十字架
        int cross_size = 2;  // 十字架长度
        // 水平线
        cv::line(img, 
                 cv::Point2d(pt.x - cross_size, pt.y),
                 cv::Point2d(pt.x + cross_size, pt.y),
                 color, 1);
        // 垂直线
        cv::line(img,
                 cv::Point2d(pt.x, pt.y - cross_size),
                 cv::Point2d(pt.x, pt.y + cross_size),
                 color, 1);
        
        // 绘制ID（如果角点数量不太多）
        // if (corners.size() < 200) {
        //     cv::putText(img, to_string(ids[i]), 
        //                cv::Point2d(pt.x + 5, pt.y - 5),
        //                cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        // }
    }
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    // 解析命令行参数
    string imageFolder;
    int tagRows, tagCols, blackBorder;
    double tagSize, tagSpacing;
    bool showVideo;
    string outputFile;
    
    if (!parseArguments(argc, argv, imageFolder, tagRows, tagCols, 
                       tagSize, tagSpacing, blackBorder, showVideo, outputFile)) {
        return 1;
    }

    // 获取所有图片文件
    vector<string> imageFiles = getImageFiles(imageFolder);
    if (imageFiles.empty()) {
        cerr << "错误: 在文件夹中未找到图片文件: " << imageFolder << endl;
        return 1;
    }

    cout << "找到 " << imageFiles.size() << " 张图片" << endl;
    cout << "AprilTag配置: " << tagRows << "x" << tagCols 
         << ", 尺寸: " << tagSize << "m, 间距: " << tagSpacing 
         << ", 黑边: " << blackBorder << endl;

    // 创建Apriltag检测器
    int numTags = tagRows * tagCols;
    basalt::ApriltagDetector detector(numTags);

    // 打开输出文件
    ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        cerr << "错误: 无法打开输出文件: " << outputFile << endl;
        return 1;
    }

    outFile << "# AprilTag检测结果\n";
    outFile << "# 格式: 图片路径 | 有效角点数 | 被拒绝角点数 | 检测到的Tag数\n";

    // 统计信息
    int totalGoodCorners = 0;
    int totalRejectedCorners = 0;
    int totalDetectedTags = 0;
    int successImages = 0;

    // 处理每张图片
    for (size_t idx = 0; idx < imageFiles.size(); idx++) {
        const string& imagePath = imageFiles[idx];
        string imageName = fs::path(imagePath).filename().string();
        
        cout << "\n处理 [" << (idx + 1) << "/" << imageFiles.size() 
             << "] " << imageName << " ... ";

        // 读取图像
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            cerr << "错误: 无法读取图片: " << imagePath << endl;
            continue;
        }

        // 转换为ManagedImage
        basalt::ManagedImage<uint16_t> managedImg = cvMatToManagedImage(img);

        // 检测角点
        Eigen::aligned_vector<Eigen::Vector2d> corners, cornersRejected;
        std::vector<int> ids, idsRejected;
        std::vector<double> radii, radiiRejected;

        detector.detectTags(managedImg, corners, ids, radii,
                           cornersRejected, idsRejected, radiiRejected);

        // 统计Tag数量（每个Tag有4个角点）
        int detectedTags = corners.size() / 4;
        
        totalGoodCorners += corners.size();
        totalRejectedCorners += cornersRejected.size();
        totalDetectedTags += detectedTags;
        
        if (corners.size() > 0) {
            successImages++;
        }

        cout << "检测到 " << corners.size() << " 个有效角点, "
             << cornersRejected.size() << " 个被拒绝角点, "
             << detectedTags << " 个Tag";

        // 写入输出文件
        outFile << imagePath << " | " << corners.size() << " | "
                << cornersRejected.size() << " | " << detectedTags << "\n";

        // 显示结果
        if (showVideo) {
            cv::Mat displayImg = img.clone();
            
            // 转换为BGR格式用于显示
            if (displayImg.channels() == 1) {
                cv::cvtColor(displayImg, displayImg, cv::COLOR_GRAY2BGR);
            }

            // 绘制有效角点（红色）
            drawCorners(displayImg, corners, ids, radii, cv::Scalar(0, 0, 255));
            
            // 绘制被拒绝的角点（蓝色）
            drawCorners(displayImg, cornersRejected, idsRejected, radiiRejected, 
                       cv::Scalar(255, 0, 0));

            // 添加文本信息
            string info = "Good: " + to_string(corners.size()) + 
                         " | Rejected: " + to_string(cornersRejected.size()) +
                         " | Tags: " + to_string(detectedTags);
            cv::putText(displayImg, info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            cv::putText(displayImg, imageName, cv::Point(10, displayImg.rows - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

            // 显示图像
            // cv::imshow("AprilTag Detection", displayImg);
            cv::imwrite(imageName, displayImg);
            // 等待按键（ESC退出，空格下一张，其他键继续）
            // int key = cv::waitKey(0) & 0xFF;
            // if (key == 27) {  // ESC
            //     cout << "\n用户中断" << endl;
            //     break;
        }
    }

    // 打印统计信息
    cout << "\n\n========== 检测统计 ==========" << endl;
    cout << "总图片数: " << imageFiles.size() << endl;
    cout << "成功检测图片数: " << successImages << endl;
    cout << "总有效角点数: " << totalGoodCorners << endl;
    cout << "总被拒绝角点数: " << totalRejectedCorners << endl;
    cout << "总检测Tag数: " << totalDetectedTags << endl;
    cout << "平均每张图片Tag数: " 
         << (successImages > 0 ? (double)totalDetectedTags / successImages : 0) << endl;
    cout << "结果已保存到: " << outputFile << endl;
    cout << "===============================" << endl;

    outFile.close();
    
    if (showVideo) {
        cv::destroyAllWindows();
    }

    return 0;
}

