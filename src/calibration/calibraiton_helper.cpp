/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/calibration/calibration_helper.h>

#include <basalt/utils/apriltag.h>

#include <tbb/parallel_for.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>

#include <opengv/sac/Ransac.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#pragma GCC diagnostic pop

#include <opencv2/calib3d/calib3d.hpp>

namespace basalt {

template <class CamT>
bool estimateTransformation(
    const CamT &cam_calib,
    const Eigen::aligned_vector<Eigen::Vector2d> &corners,
    const std::vector<int> &corner_ids,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    Sophus::SE3d &T_target_camera, size_t &num_inliers) {
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (size_t i = 0; i < corners.size(); i++) {
    Eigen::Vector4d tmp;
    if (!cam_calib.unproject(corners[i], tmp)) {
      continue;
    }
    Eigen::Vector3d bearing = tmp.head<3>();
    Eigen::Vector3d point = aprilgrid_corner_pos_3d[corner_ids[i]].head<3>();
    bearing.normalize();

    bearingVectors.push_back(bearing);
    points.push_back(point);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 1.0 - cos(atan(sqrt(2.0) * 1 / cam_calib.getParam()[0]));
  std::cout << "cam_calib.getParam()[0]: " << cam_calib.getParam()[0] << " threshold: " << ransac.threshold_ << std::endl;
  ransac.max_iterations_ = 50;

  ransac.computeModel();

  T_target_camera =
      Sophus::SE3d(ransac.model_coefficients_.topLeftCorner<3, 3>(),
                   ransac.model_coefficients_.topRightCorner<3, 1>());

  num_inliers = ransac.inliers_.size();

  return ransac.inliers_.size() > 8;
}

void CalibHelper::detectCorners(const VioDatasetPtr &vio_data,
                                const AprilGrid &april_grid,
                                CalibCornerMap &calib_corners,
                                CalibCornerMap &calib_corners_rejected) {
  calib_corners.clear();
  calib_corners_rejected.clear();

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, vio_data->get_image_timestamps().size()),
      [&](const tbb::blocked_range<size_t> &r) {
        const int numTags = april_grid.getTagCols() * april_grid.getTagRows();
        ApriltagDetector ad(numTags);

        for (size_t j = r.begin(); j != r.end(); ++j) {
          int64_t timestamp_ns = vio_data->get_image_timestamps()[j];
          const std::vector<ImageData> &img_vec =
              vio_data->get_image_data(timestamp_ns);

          for (size_t i = 0; i < img_vec.size(); i++) {
            if (img_vec[i].img.get()) {
              CalibCornerData ccd_good;
              CalibCornerData ccd_bad;
              ad.detectTags(*img_vec[i].img, ccd_good.corners,
                            ccd_good.corner_ids, ccd_good.radii,
                            ccd_bad.corners, ccd_bad.corner_ids, ccd_bad.radii);

              //                std::cout << "image (" << timestamp_ns << ","
              //                << i
              //                          << ")  detected " <<
              //                          ccd_good.corners.size()
              //                          << "corners (" <<
              //                          ccd_bad.corners.size()
              //                          << " rejected)" << std::endl;

              TimeCamId tcid(timestamp_ns, i);

              calib_corners.emplace(tcid, ccd_good);
              calib_corners_rejected.emplace(tcid, ccd_bad);
            }
          }
        }
      });
}

void CalibHelper::initCamPoses(
    const Calibration<double>::Ptr &calib,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    CalibCornerMap &calib_corners, CalibInitPoseMap &calib_init_poses) {
  calib_init_poses.clear();

  std::vector<TimeCamId> corners;
  corners.reserve(calib_corners.size());
  for (const auto &kv : calib_corners) {
    corners.emplace_back(kv.first);
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, corners.size()),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t j = r.begin(); j != r.end(); ++j) {
                        TimeCamId tcid = corners[j];
                        const CalibCornerData &ccd = calib_corners.at(tcid);

                        CalibInitPoseData cp;

                        computeInitialPose(calib, tcid.cam_id,
                                           aprilgrid_corner_pos_3d, ccd, cp);

                        calib_init_poses.emplace(tcid, cp);
                      }
                    });

  // Print statistics for each frame after pose computation
  std::cout << "\n[INIT POSE] Frame statistics (total_corners, num_inliers):" << std::endl;
  for (const auto &tcid : corners) {
    const CalibCornerData &ccd = calib_corners.at(tcid);
    size_t total_corners = ccd.corners.size();
    
    auto it = calib_init_poses.find(tcid);
    size_t num_inliers = 0;
    if (it != calib_init_poses.end()) {
      num_inliers = it->second.num_inliers;
    }
    
    std::cout << "  timestamp_ns: " << tcid.frame_id 
              << " cam_id: " << tcid.cam_id
              << " total_corners: " << total_corners
              << " num_inliers: " << num_inliers;
    
    if (num_inliers == 0) {
      std::cout << " (failed)";
    } else if (num_inliers < 8) {
      std::cout << " (too few corners)";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/// @brief 通过单张 AprilGrid 观测初始化通用相机内参
///
/// 该函数假设图像中心已知（取为图像中心），在图像中搜索一条“非径向”的
/// AprilGrid 直线，用于初始化等距广角类相机模型的焦距（如 EUCM、DS、KB4 等）。
/// 实现上基于文献中对等距成像的近似，将选取的角点构造成矩阵，使用 SVD
/// 求解出一条最优成像直线，再由此反推出有效焦距。
///
/// 具体步骤：
/// 1. 将像素坐标减去图像中心，构造每个角点的 \f$[u, v, 0.5, -0.5(u^2+v^2)]\f$
///    四维向量，并按行/列遍历 AprilGrid 的角点组合成矩阵 \f$P\f$；
/// 2. 对 \f$P\f$ 做 SVD，取最小奇异值对应的右奇异向量 \f$C\f$ 作为成像直线参数；
/// 3. 过滤掉近似径向的直线，仅保留与光轴有足够夹角的候选直线；
/// 4. 对每条候选直线估计焦距 \f$\gamma\f$，构造一个临时的
///    `UnifiedCamera` 模型并估计位姿与重投影误差；
/// 5. 选择重投影误差最小的候选，得到最终的 \f$\gamma_0\f$，并设置
///    \f$[f_x, f_y, c_x, c_y] = [\gamma_0/2, \gamma_0/2, c_u, c_v]\f$ 作为初始内参。
///
/// 该初始化主要用于非针孔模型的第一阶段粗初始化，后续会在整体优化中联合精调。
///
/// @param[in]  corners     图像中检测到的 AprilGrid 角点像素坐标（单位：像素）
/// @param[in]  corner_ids  每个角点对应的 AprilGrid 角点 ID（索引到 3D 角点坐标）
/// @param[in]  aprilgrid   AprilGrid 标定板配置，包含 3D 角点坐标和网格尺寸
/// @param[in]  cols        图像宽度（像素）
/// @param[in]  rows        图像高度（像素）
/// @param[out] init_intr   输出的初始内参向量 \f$[f_x, f_y, c_x, c_y]^T\f$
///
/// @return 如果成功找到合适的非径向直线并得到稳定的初始焦距，则返回 true；
///         否则返回 false，不修改 @p init_intr。
bool CalibHelper::initializeIntrinsics(
    const Eigen::aligned_vector<Eigen::Vector2d> &corners,
    const std::vector<int> &corner_ids, const AprilGrid &aprilgrid, int cols,
    int rows, Eigen::Vector4d &init_intr) {
  // 步骤 1：建立角点 ID 到像素坐标的映射，便于快速查找
  Eigen::aligned_map<int, Eigen::Vector2d> id_to_corner;
  for (size_t i = 0; i < corner_ids.size(); i++) {
    id_to_corner[corner_ids[i]] = corners[i];
  }

  // 步骤 2：初始化参数
  // _xi: Unified Camera Model 的投影参数（固定为 1.0）
  const double _xi = 1.0;
  // _cu, _cv: 假设图像中心就是主点位置（像素坐标从 0 开始，所以减 0.5）
  const double _cu = cols / 2.0 - 0.5;
  const double _cv = rows / 2.0 - 0.5;

  // 步骤 3：初始化用于存储最佳结果的变量
  double gamma0 = 0.0;  // 最佳焦距参数
  double minReprojErr = std::numeric_limits<double>::max();  // 最小重投影误差

  // 步骤 4：获取 AprilGrid 的尺寸信息
  const size_t target_cols = aprilgrid.getTagCols();  // 标定板的列数（tag 数量）
  const size_t target_rows = aprilgrid.getTagRows();  // 标定板的行数（tag 数量）

  bool success = false;
  // 步骤 5：遍历两种角点偏移模式（每个 tag 有 4 个角点，可以取不同的起始角点）
  for (int tag_corner_offset = 0; tag_corner_offset < 2; tag_corner_offset++)
    // 步骤 6：遍历 AprilGrid 的每一行（水平方向的直线）
    for (size_t r = 0; r < target_rows; ++r) {
      // 步骤 7：为当前行收集角点数据，构造矩阵 P
      // P 的每一列是一个角点的四维向量 [u, v, 0.5, -0.5*(u^2+v^2)]
      Eigen::aligned_vector<Eigen::Vector4d> P;

      // 遍历当前行的每一列
      for (size_t c = 0; c < target_cols; ++c) {
        // 计算当前 tag 的角点 ID 偏移量（每个 tag 有 4 个角点）
        int tag_offset = (r * target_cols + c) << 2;

        // 每个 tag 取 2 个角点（形成一条边）
        for (int i = 0; i < 2; i++) {
          int corner_id = tag_offset + i + tag_corner_offset * 2;

          // 如果该角点在检测结果中存在
          if (id_to_corner.find(corner_id) != id_to_corner.end()) {
            const Eigen::Vector2d imagePoint = id_to_corner[corner_id];

            // 步骤 8：将像素坐标转换为以图像中心为原点的坐标
            double u = imagePoint[0] - _cu;  // 水平方向偏移
            double v = imagePoint[1] - _cv;  // 垂直方向偏移

            // 步骤 9：构造四维向量，用于等距投影模型的直线拟合
            // 这个形式基于等距投影的数学特性：直线在图像中的投影满足特定约束
            P.emplace_back(u, v, 0.5, -0.5 * (square(u) + square(v)));
          }
        }
      }

      // 步骤 10：检查是否有足够的角点（至少 8 个）用于拟合直线
      const int MIN_CORNERS = 8;
      if (P.size() > MIN_CORNERS) {
        // 步骤 11：将向量数组转换为矩阵形式（4 行 N 列，N 为角点数量）
        Eigen::Map<Eigen::Matrix4Xd> P_mat((double *)P.data(), 4, P.size());

        // 步骤 12：转置矩阵，准备进行 SVD 分解
        // P_mat_t 是 N×4 矩阵，每一行是一个角点的四维向量
        Eigen::MatrixXd P_mat_t = P_mat.transpose();

        // 步骤 13：对矩阵进行奇异值分解（SVD）
        // 目的是找到使所有点满足直线约束的参数向量 C
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            P_mat_t, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // 步骤 14：取最小奇异值对应的右奇异向量作为直线参数
        // C 是 4 维向量，表示拟合出的直线参数
        Eigen::Vector4d C = svd.matrixV().col(3);

        // 步骤 15：检查参数的有效性
        // t = C(0)^2 + C(1)^2 + C(2)*C(3)，必须为正数
        double t = square(C(0)) + square(C(1)) + C(2) * C(3);
        if (t < 0) {
          continue;  // 无效参数，跳过
        }

        // 步骤 16：检查该直线是否为非径向直线（不通过图像中心）
        // 计算归一化的法向量分量
        double d = sqrt(1.0 / t);
        double nx = C(0) * d;  // 法向量 x 分量
        double ny = C(1) * d;  // 法向量 y 分量
        
        // 如果 hypot(nx, ny) > 0.95，说明直线接近径向（通过图像中心）
        // 径向直线对焦距估计不敏感，需要过滤掉
        if (hypot(nx, ny) > 0.95) {
          continue;  // 跳过径向直线
        }

        // 步骤 17：计算法向量的 z 分量（归一化）
        double nz = sqrt(1.0 - square(nx) - square(ny));
        
        // 步骤 18：从直线参数反推出焦距参数 gamma
        // gamma 是等距投影模型中的有效焦距
        double gamma = fabs(C(2) * d / nz);

        // 步骤 19：构造临时的 UnifiedCamera 模型用于验证
        // 参数：fx=fy=gamma/2, cx=cu, cy=cv, xi=0.5
        Eigen::Matrix<double, 5, 1> calib;
        calib << 0.5 * gamma, 0.5 * gamma, _cu, _cv, 0.5 * _xi;

        UnifiedCamera<double> cam_calib(calib);

        // 步骤 20：使用临时相机模型估计标定板到相机的位姿变换
        size_t num_inliers;
        Sophus::SE3d T_target_camera;
        if (!estimateTransformation(cam_calib, corners, corner_ids,
                                    aprilgrid.aprilgrid_corner_pos_3d,
                                    T_target_camera, num_inliers)) {
          continue;  // 位姿估计失败，跳过
        }

        // 步骤 21：计算重投影误差，评估当前焦距估计的质量
        double reprojErr = 0.0;
        size_t numReprojected = computeReprojectionError(
            cam_calib, corners, corner_ids, aprilgrid.aprilgrid_corner_pos_3d,
            T_target_camera, reprojErr);

        // 步骤 22：如果重投影误差足够小，更新最佳结果
        if (numReprojected > MIN_CORNERS) {
          double avgReprojErr = reprojErr / numReprojected;

          // 选择重投影误差最小的候选作为最终结果
          if (avgReprojErr < minReprojErr) {
            minReprojErr = avgReprojErr;
            gamma0 = gamma;  // 保存最佳焦距
            success = true;
          }
        }

      }  // 如果该行有足够的角点
    }    // 遍历每一行

  // 步骤 23：如果找到合适的非径向直线，设置初始内参
  // fx = fy = gamma0/2（等距投影模型），cx = cu, cy = cv
  if (success) init_intr << 0.5 * gamma0, 0.5 * gamma0, _cu, _cv;

  return success;
}

/// @brief 使用 Zhang 标定方法初始化理想针孔相机内参
///
/// 该函数实现了 Zhang 经典标定算法，用多张 AprilGrid 图像估计针孔相机的
/// 焦距和主点。假设图像中心接近主点位置，首先固定主点为图像中心，然后
/// 对每一张图像估计平面单应性矩阵，再通过线性最小二乘求解焦距。
///
/// 主要步骤：
/// 1. 对每张输入图像，构造世界平面点集（AprilGrid 角点在标定板平面上的 2D 坐标）
///    与像素平面点集（检测到的角点像素坐标），使用 `cv::findHomography` 估计单应性 H；
/// 2. 将 H 平移到以图像中心为原点的坐标系，计算列向量 h、v 及其和/差 d1、d2；
/// 3. 对每张图像构造两行线性方程，堆叠得到线性系统 \f$A f = b\f$；
/// 4. 通过 \f$(A^T A)^{-1} A^T b\f$ 求解 \f$f = [1/f_x^2, 1/f_y^2]^T\f$，
///    再取平方根得到 \f$f_x, f_y\f$；
/// 5. 最终将 \f$[f_x, f_y, c_x, c_y]\f$ 写入 @p init_intr 作为针孔模型的初始内参。
///
/// @param[in]  pinhole_corners  若干张图像的角点检测结果，每个元素对应一帧
/// @param[in]  aprilgrid        AprilGrid 标定板配置，提供 3D/2D 角点位置
/// @param[in]  cols             图像宽度（像素）
/// @param[in]  rows             图像高度（像素）
/// @param[out] init_intr        输出的初始针孔内参 \f$[f_x, f_y, c_x, c_y]^T\f$
///
/// @return 如果所有输入图像的单应性都成功估计且线性系统可解，返回 true；
///         若任意图像单应性估计失败，则立即返回 false，不保证 @p init_intr 有效。
bool CalibHelper::initializeIntrinsicsPinhole(
    const std::vector<CalibCornerData *> pinhole_corners,
    const AprilGrid &aprilgrid, int cols, int rows,
    Eigen::Vector4d &init_intr) {
  // 步骤 1：假设图像中心就是主点位置
  // 这是 Zhang 方法的基本假设，对于大多数相机是合理的近似
  const double _cu = cols / 2.0 - 0.5;  // 主点 x 坐标（像素坐标从 0 开始）
  const double _cv = rows / 2.0 - 0.5;  // 主点 y 坐标

  // 参考：Z. Zhang, A Flexible New Technique for Camera Calibration, PAMI 2000

  // 步骤 2：初始化线性系统
  // 每张图像贡献 2 个方程，共 nImages 张图像，所以 A 是 2*nImages × 2 矩阵
  // 求解的未知数是 f = [1/fx^2, 1/fy^2]^T
  size_t nImages = pinhole_corners.size();

  Eigen::MatrixXd A(nImages * 2, 2);  // 系数矩阵
  Eigen::VectorXd b(nImages * 2, 1);  // 常数项向量

  int i = 0;  // 当前图像索引

  // 步骤 3：对每张图像处理
  for (const CalibCornerData *ccd : pinhole_corners) {
    const auto &corners = ccd->corners;      // 该图像的角点像素坐标
    const auto &corner_ids = ccd->corner_ids; // 对应的角点 ID

    // 步骤 4：准备单应性估计的输入数据
    // M: 标定板平面上的 2D 坐标（AprilGrid 角点在标定板平面上的 X, Y 坐标）
    // imagePoints: 图像中检测到的角点像素坐标
    std::vector<cv::Point2f> M(corners.size()), imagePoints(corners.size());
    for (size_t j = 0; j < corners.size(); ++j) {
      // 提取标定板平面上的 2D 坐标（忽略 Z 坐标，因为标定板是平面）
      M.at(j) =
          cv::Point2f(aprilgrid.aprilgrid_corner_pos_3d[corner_ids[j]][0],
                      aprilgrid.aprilgrid_corner_pos_3d[corner_ids[j]][1]);

      // 提取图像中的像素坐标
      imagePoints.at(j) = cv::Point2f(corners[j][0], corners[j][1]);
    }

    // 步骤 5：使用 OpenCV 估计单应性矩阵 H
    // H 将标定板平面坐标映射到图像像素坐标：imagePoint = H * M
    cv::Mat H = cv::findHomography(M, imagePoints);

    // 如果单应性估计失败，立即返回 false
    if (H.empty()) return false;

    // 步骤 6：将单应性矩阵平移到以图像中心为原点的坐标系
    // 这是为了利用 Zhang 方法中的约束条件
    // 平移变换：H' = H - [cx; cy; 0] * H 的第三行
    H.at<double>(0, 0) -= H.at<double>(2, 0) * _cu;
    H.at<double>(0, 1) -= H.at<double>(2, 1) * _cu;
    H.at<double>(0, 2) -= H.at<double>(2, 2) * _cu;
    H.at<double>(1, 0) -= H.at<double>(2, 0) * _cv;
    H.at<double>(1, 1) -= H.at<double>(2, 1) * _cv;
    H.at<double>(1, 2) -= H.at<double>(2, 2) * _cv;

    // 步骤 7：提取单应性矩阵的列向量
    // h: H 的第一列（水平方向）
    // v: H 的第二列（垂直方向）
    double h[3], v[3], d1[3], d2[3];
    double n[4] = {0, 0, 0, 0};  // 用于归一化的范数

    for (int j = 0; j < 3; ++j) {
      double t0 = H.at<double>(j, 0);  // H 的第一列
      double t1 = H.at<double>(j, 1);  // H 的第二列
      h[j] = t0;
      v[j] = t1;
      // d1, d2: h 和 v 的和与差，用于构造约束方程
      d1[j] = (t0 + t1) * 0.5;  // (h + v) / 2
      d2[j] = (t0 - t1) * 0.5;  // (h - v) / 2
      // 计算各向量的平方和，用于后续归一化
      n[0] += t0 * t0;  // ||h||^2
      n[1] += t1 * t1;  // ||v||^2
      n[2] += d1[j] * d1[j];  // ||d1||^2
      n[3] += d2[j] * d2[j];  // ||d2||^2
    }

    // 步骤 8：计算归一化因子（各向量的范数）
    for (int j = 0; j < 4; ++j) {
      n[j] = 1.0 / sqrt(n[j]);  // 1 / ||vector||
    }

    // 步骤 9：归一化向量 h, v, d1, d2
    for (int j = 0; j < 3; ++j) {
      h[j] *= n[0];   // 归一化 h
      v[j] *= n[1];   // 归一化 v
      d1[j] *= n[2];  // 归一化 d1
      d2[j] *= n[3];  // 归一化 d2
    }

    // 步骤 10：根据 Zhang 方法的约束条件构造线性方程
    // 对于针孔相机，单应性矩阵的列向量满足以下约束：
    // h[0]*v[0] / fx^2 + h[1]*v[1] / fy^2 = -h[2]*v[2]
    // d1[0]*d2[0] / fx^2 + d1[1]*d2[1] / fy^2 = -d1[2]*d2[2]
    // 设 f = [1/fx^2, 1/fy^2]^T，则上述约束可写成 A * f = b
    A(i * 2, 0) = h[0] * v[0];      // 第一个方程的 fx^2 系数
    A(i * 2, 1) = h[1] * v[1];      // 第一个方程的 fy^2 系数
    A(i * 2 + 1, 0) = d1[0] * d2[0]; // 第二个方程的 fx^2 系数
    A(i * 2 + 1, 1) = d1[1] * d2[1]; // 第二个方程的 fy^2 系数
    b(i * 2, 0) = -h[2] * v[2];     // 第一个方程的常数项
    b(i * 2 + 1, 0) = -d1[2] * d2[2]; // 第二个方程的常数项

    i++;  // 移动到下一张图像
  }

  // 步骤 11：求解线性最小二乘问题 A * f = b
  // 使用 LDLT 分解求解：f = (A^T * A)^(-1) * A^T * b
  // f = [1/fx^2, 1/fy^2]^T
  Eigen::Vector2d f = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  // 步骤 12：从 f 中恢复焦距 fx 和 fy
  // fx = sqrt(1 / f[0]), fy = sqrt(1 / f[1])
  double fx = sqrt(fabs(1.0 / f(0)));
  double fy = sqrt(fabs(1.0 / f(1)));

  // 步骤 13：设置最终的内参向量 [fx, fy, cx, cy]
  init_intr << fx, fy, _cu, _cv;

  return true;
}

void CalibHelper::computeInitialPose(
    const Calibration<double>::Ptr &calib, size_t cam_id,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    const CalibCornerData &cd, CalibInitPoseData &cp) {
  if (cd.corners.size() < 8) {
    cp.num_inliers = 0;
    return;
  }

  bool success;
  size_t num_inliers;

  std::visit(
      [&](const auto &cam) {
        Sophus::SE3d T_target_camera;
        success = estimateTransformation(cam, cd.corners, cd.corner_ids,
                                         aprilgrid_corner_pos_3d, cp.T_a_c,
                                         num_inliers);
      },
      calib->intrinsics[cam_id].variant);

  if (success) {
    // std::cout << "size " << cd.corners.size() << " num_inliers " << num_inliers << std::endl;
    Eigen::Matrix4d T_c_a_init = cp.T_a_c.inverse().matrix();

    std::vector<bool> proj_success;
    calib->intrinsics[cam_id].project(aprilgrid_corner_pos_3d, T_c_a_init,
                                      cp.reprojected_corners, proj_success);

    cp.num_inliers = num_inliers;
  } else {
    cp.num_inliers = 0;
  }
}

size_t CalibHelper::computeReprojectionError(
    const UnifiedCamera<double> &cam_calib,
    const Eigen::aligned_vector<Eigen::Vector2d> &corners,
    const std::vector<int> &corner_ids,
    const Eigen::aligned_vector<Eigen::Vector4d> &aprilgrid_corner_pos_3d,
    const Sophus::SE3d &T_target_camera, double &error) {
  size_t num_projected = 0;
  error = 0;

  Eigen::Matrix4d T_camera_target = T_target_camera.inverse().matrix();

  for (size_t i = 0; i < corners.size(); i++) {
    Eigen::Vector4d p_cam =
        T_camera_target * aprilgrid_corner_pos_3d[corner_ids[i]];
    Eigen::Vector2d res;
    cam_calib.project(p_cam, res);
    res -= corners[i];

    num_projected++;
    error += res.norm();
  }

  return num_projected;
}
}  // namespace basalt
