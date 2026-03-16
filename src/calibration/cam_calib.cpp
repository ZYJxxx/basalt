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

#include <basalt/calibration/cam_calib.h>

#include <basalt/utils/system_utils.h>

#include <basalt/calibration/vignette.h>

#include <basalt/optimization/poses_optimize.h>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/filesystem.h>

#include <pangolin/display/default_font.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <unordered_set>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#include <direct.h>
#define getcwd _getcwd
#endif

#include <cmath>

namespace basalt {

CamCalib::CamCalib(const std::string &dataset_path,
                   const std::string &dataset_type,
                   const std::string &aprilgrid_path,
                   const std::string &cache_path,
                   const std::string &cache_dataset_name, int skip_images,
                   const std::vector<std::string> &cam_types, bool show_gui)
    : dataset_path(dataset_path),
      dataset_type(dataset_type),
      april_grid(aprilgrid_path),
      cache_path(ensure_trailing_slash(cache_path)),
      cache_dataset_name(cache_dataset_name),
      skip_images(skip_images),
      cam_types(cam_types),
      show_gui(show_gui),
      show_frame("ui.show_frame", 0, 0, 1500),
      show_corners("ui.show_corners", true, false, true),
      show_corners_rejected("ui.show_corners_rejected", false, false, true),
      show_init_reproj("ui.show_init_reproj", false, false, true),
      show_opt("ui.show_opt", true, false, true),
      show_vign("ui.show_vign", false, false, true),
      show_ids("ui.show_ids", false, false, true),
      huber_thresh("ui.huber_thresh", 4.0, 0.1, 10.0),
      opt_intr("ui.opt_intr", true, false, true),
      opt_until_convg("ui.opt_until_converge", false, false, true),
      stop_thresh("ui.stop_thresh", 1e-8, 1e-10, 0.01, true) {
  if (show_gui) initGui();

  if (!fs::exists(cache_path)) {
    fs::create_directory(cache_path);
  }

  pangolin::ColourWheel cw;
  for (int i = 0; i < 20; i++) {
    cam_colors.emplace_back(cw.GetUniqueColour());
  }
}

CamCalib::~CamCalib() {
  if (processing_thread) {
    processing_thread->join();
  }
}

void CamCalib::initGui() {
  pangolin::CreateWindowAndBind("Main", 1600, 1000);

  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                        pangolin::Attach::Pix(UI_WIDTH));

  img_view_display =
      &pangolin::CreateDisplay()
           .SetBounds(0.5, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
           .SetLayout(pangolin::LayoutEqual);

  pangolin::View &vign_plot_display =
      pangolin::CreateDisplay().SetBounds(0.0, 0.5, 0.72, 1.0);

  vign_plotter.reset(new pangolin::Plotter(&vign_data_log, 0.0, 1000.0, 0.0,
                                           1.0, 0.01f, 0.01f));
  vign_plot_display.AddDisplay(*vign_plotter);

  pangolin::View &polar_error_display = pangolin::CreateDisplay().SetBounds(
      0.0, 0.5, pangolin::Attach::Pix(UI_WIDTH), 0.43);

  polar_plotter.reset(
      new pangolin::Plotter(nullptr, 0.0, 120.0, 0.0, 1.0, 0.01f, 0.01f));
  polar_error_display.AddDisplay(*polar_plotter);

  pangolin::View &azimuthal_plot_display =
      pangolin::CreateDisplay().SetBounds(0.0, 0.5, 0.45, 0.7);

  azimuth_plotter.reset(
      new pangolin::Plotter(nullptr, -180.0, 180.0, 0.0, 1.0, 0.01f, 0.01f));
  azimuthal_plot_display.AddDisplay(*azimuth_plotter);

  pangolin::Var<std::function<void(void)>> load_dataset(
      "ui.load_dataset", std::bind(&CamCalib::loadDataset, this));

  pangolin::Var<std::function<void(void)>> detect_corners(
      "ui.detect_corners", std::bind(&CamCalib::detectCorners, this));

  pangolin::Var<std::function<void(void)>> init_cam_intrinsics(
      "ui.init_cam_intr", std::bind(&CamCalib::initCamIntrinsics, this));

  pangolin::Var<std::function<void(void)>> init_cam_poses(
      "ui.init_cam_poses", std::bind(&CamCalib::initCamPoses, this));

  pangolin::Var<std::function<void(void)>> init_cam_extrinsics(
      "ui.init_cam_extr", std::bind(&CamCalib::initCamExtrinsics, this));

  pangolin::Var<std::function<void(void)>> init_opt(
      "ui.init_opt", std::bind(&CamCalib::initOptimization, this));

  pangolin::Var<std::function<void(void)>> optimize(
      "ui.optimize", std::bind(&CamCalib::optimize, this));

  pangolin::Var<std::function<void(void)>> save_calib(
      "ui.save_calib", std::bind(&CamCalib::saveCalib, this));

  pangolin::Var<std::function<void(void)>> compute_vign(
      "ui.compute_vign", std::bind(&CamCalib::computeVign, this));

  pangolin::Var<std::function<void(void)>> save_corner_images(
      "ui.save_corner_images", std::bind(&CamCalib::saveCornerImages, this));

  setNumCameras(1);
}

void CamCalib::computeVign() {
  Eigen::aligned_vector<Eigen::Vector2d> optical_centers;
  for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
    optical_centers.emplace_back(
        calib_opt->calib->intrinsics[i].getParam().segment<2>(2));
  }

  std::map<TimeCamId, Eigen::aligned_vector<Eigen::Vector3d>>
      reprojected_vignette2;
  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
    const std::vector<ImageData> img_vec =
        vio_dataset->get_image_data(timestamp_ns);

    for (size_t j = 0; j < calib_opt->calib->intrinsics.size(); j++) {
      TimeCamId tcid(timestamp_ns, j);

      auto it = reprojected_vignette.find(tcid);

      if (it != reprojected_vignette.end() && img_vec[j].img.get()) {
        Eigen::aligned_vector<Eigen::Vector3d> rv;
        rv.resize(it->second.corners_proj.size());

        for (size_t k = 0; k < it->second.corners_proj.size(); k++) {
          Eigen::Vector2d pos = it->second.corners_proj[k];

          rv[k].head<2>() = pos;

          if (img_vec[j].img->InBounds(pos[0], pos[1], 1) &&
              it->second.corners_proj_success[k]) {
            double val = img_vec[j].img->interp(pos);
            val /= std::numeric_limits<uint16_t>::max();

            if (img_vec[j].exposure > 0) {
              val *= 0.001 / img_vec[j].exposure;  // bring to common exposure
            }

            rv[k][2] = val;
          } else {
            rv[k][2] = -1;
          }
        }

        reprojected_vignette2.emplace(tcid, rv);
      }
    }
  }

  VignetteEstimator ve(vio_dataset, optical_centers,
                       calib_opt->calib->resolution, reprojected_vignette2,
                       april_grid);

  ve.optimize();
  ve.compute_error(&reprojected_vignette_error);

  std::vector<std::vector<float>> vign_data;
  ve.compute_data_log(vign_data);
  vign_data_log.Clear();
  for (const auto &v : vign_data) vign_data_log.Log(v);

  {
    vign_plotter->ClearSeries();
    vign_plotter->ClearMarkers();

    for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
      vign_plotter->AddSeries("$i", "$" + std::to_string(2 * i),
                              pangolin::DrawingModeLine, cam_colors[i],
                              "vignette camera " + std::to_string(i));
    }

    vign_plotter->ScaleViewSmooth(vign_data_log.Samples() / 1000.0f, 1.0f, 0.0f,
                                  0.5f);
  }

  ve.save_vign_png(cache_path);

  calib_opt->setVignette(ve.get_vign_param());

  std::cout << "Saved vignette png files to " << cache_path << std::endl;
}

void CamCalib::setNumCameras(size_t n) {
  while (img_view.size() < n && show_gui) {
    std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

    size_t idx = img_view.size();
    img_view.push_back(iv);

    img_view_display->AddDisplay(*iv);
    iv->extern_draw_function = std::bind(&CamCalib::drawImageOverlay, this,
                                         std::placeholders::_1, idx);
  }
}

void CamCalib::renderingLoop() {
  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (vio_dataset.get()) {
      if (show_frame.GuiChanged()) {
        size_t frame_id = static_cast<size_t>(show_frame);
        int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

        const std::vector<ImageData> &img_vec =
            vio_dataset->get_image_data(timestamp);

        for (size_t cam_id = 0; cam_id < vio_dataset->get_num_cams(); cam_id++)
          if (img_vec[cam_id].img.get()) {
            pangolin::GlPixFormat fmt;
            fmt.glformat = GL_LUMINANCE;
            fmt.gltype = GL_UNSIGNED_SHORT;
            fmt.scalable_internal_format = GL_LUMINANCE16;

            img_view[cam_id]->SetImage(
                img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
          } else {
            img_view[cam_id]->Clear();
          }
      }
    }

    if (opt_until_convg) {
      bool converged = optimizeWithParam(true);
      if (converged) opt_until_convg = false;
    }

    pangolin::FinishFrame();
  }
}

void CamCalib::computeProjections() {
  reprojected_corners.clear();
  reprojected_vignette.clear();

  if (!calib_opt.get() || !vio_dataset.get()) return;

  constexpr int ANGLE_BIN_SIZE = 2;
  std::vector<Eigen::Matrix<double, 180 / ANGLE_BIN_SIZE, 1>> polar_sum(
      calib_opt->calib->intrinsics.size());
  std::vector<Eigen::Matrix<int, 180 / ANGLE_BIN_SIZE, 1>> polar_num(
      calib_opt->calib->intrinsics.size());

  std::vector<Eigen::Matrix<double, 360 / ANGLE_BIN_SIZE, 1>> azimuth_sum(
      calib_opt->calib->intrinsics.size());
  std::vector<Eigen::Matrix<int, 360 / ANGLE_BIN_SIZE, 1>> azimuth_num(
      calib_opt->calib->intrinsics.size());

  for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
    polar_sum[i].setZero();
    polar_num[i].setZero();
    azimuth_sum[i].setZero();
    azimuth_num[i].setZero();
  }

  for (size_t j = 0; j < vio_dataset->get_image_timestamps().size(); ++j) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[j];

    for (size_t i = 0; i < calib_opt->calib->intrinsics.size(); i++) {
      TimeCamId tcid(timestamp_ns, i);

      ProjectedCornerData rc, rv;
      Eigen::aligned_vector<Eigen::Vector2d> polar_azimuthal_angle;

      Sophus::SE3d T_c_w_ =
          (calib_opt->getT_w_i(timestamp_ns) * calib_opt->calib->T_i_c[i])
              .inverse();

      Eigen::Matrix4d T_c_w = T_c_w_.matrix();

      calib_opt->calib->intrinsics[i].project(
          april_grid.aprilgrid_corner_pos_3d, T_c_w, rc.corners_proj,
          rc.corners_proj_success, polar_azimuthal_angle);

      calib_opt->calib->intrinsics[i].project(
          april_grid.aprilgrid_vignette_pos_3d, T_c_w, rv.corners_proj,
          rv.corners_proj_success);

      reprojected_corners.emplace(tcid, rc);
      reprojected_vignette.emplace(tcid, rv);

      // Compute reprojection histogrames over polar and azimuth angle
      auto it = calib_corners.find(tcid);
      if (it != calib_corners.end()) {
        for (size_t k = 0; k < it->second.corners.size(); k++) {
          size_t id = it->second.corner_ids[k];

          if (rc.corners_proj_success[id]) {
            double error = (it->second.corners[k] - rc.corners_proj[id]).norm();

            size_t polar_bin =
                180 * polar_azimuthal_angle[id][0] / (M_PI * ANGLE_BIN_SIZE);

            polar_sum[tcid.cam_id][polar_bin] += error;
            polar_num[tcid.cam_id][polar_bin] += 1;

            size_t azimuth_bin =
                180 / ANGLE_BIN_SIZE + (180.0 * polar_azimuthal_angle[id][1]) /
                                           (M_PI * ANGLE_BIN_SIZE);

            azimuth_sum[tcid.cam_id][azimuth_bin] += error;
            azimuth_num[tcid.cam_id][azimuth_bin] += 1;
          }
        }
      }
    }
  }

  while (polar_data_log.size() < calib_opt->calib->intrinsics.size()) {
    polar_data_log.emplace_back(new pangolin::DataLog);
  }

  while (azimuth_data_log.size() < calib_opt->calib->intrinsics.size()) {
    azimuth_data_log.emplace_back(new pangolin::DataLog);
  }

  constexpr int MIN_POINTS_HIST = 3;
  polar_plotter->ClearSeries();
  azimuth_plotter->ClearSeries();

  for (size_t c = 0; c < calib_opt->calib->intrinsics.size(); c++) {
    polar_data_log[c]->Clear();
    azimuth_data_log[c]->Clear();

    // 用于记录最大重投影误差及其对应的角度
    double max_polar_error = 0.0;
    double max_polar_angle = 0.0;
    double max_azimuth_error = 0.0;
    double max_azimuth_angle = 0.0;

    // 统计用于绘制的总点数
    int total_polar_points = 0;
    int total_azimuth_points = 0;
    int total_polar_bins = 0;
    int total_azimuth_bins = 0;

    for (int i = 0; i < polar_sum[c].rows(); i++) {
      if (polar_num[c][i] > MIN_POINTS_HIST) {
        double x_coord = ANGLE_BIN_SIZE * i + ANGLE_BIN_SIZE / 2.0;
        double mean_reproj = polar_sum[c][i] / polar_num[c][i];

        // 记录最大误差和对应的角度
        if (mean_reproj > max_polar_error) {
          max_polar_error = mean_reproj;
          max_polar_angle = x_coord;
        }

        // 统计用于绘制的点数和bin数
        total_polar_points += polar_num[c][i];
        total_polar_bins++;

        polar_data_log[c]->Log(x_coord, mean_reproj);
      }
    }

    for (int i = 0; i < azimuth_sum[c].rows(); i++) {
      if (azimuth_num[c][i] > MIN_POINTS_HIST) {
        double x_coord = ANGLE_BIN_SIZE * i + ANGLE_BIN_SIZE / 2.0 - 180.0;
        double mean_reproj = azimuth_sum[c][i] / azimuth_num[c][i];

        // 记录最大误差和对应的角度
        if (mean_reproj > max_azimuth_error) {
          max_azimuth_error = mean_reproj;
          max_azimuth_angle = x_coord;
        }

        // 统计用于绘制的点数和bin数
        total_azimuth_points += azimuth_num[c][i];
        total_azimuth_bins++;

        azimuth_data_log[c]->Log(x_coord, mean_reproj);
      }
    }

    // 打印最大重投影误差信息和总点数
    if (max_polar_error > 0.0) {
      std::cout << "[Camera " << c << "] Max polar angle reprojection error: "
                << std::fixed << std::setprecision(4) << max_polar_error
                << " pixels at polar angle: " << std::setprecision(2) 
                << max_polar_angle << " degrees (total points: " 
                << total_polar_points << ", bins: " << total_polar_bins << ")" << std::endl;
    }

    if (max_azimuth_error > 0.0) {
      std::cout << "[Camera " << c << "] Max azimuth angle reprojection error: "
                << std::fixed << std::setprecision(4) << max_azimuth_error
                << " pixels at azimuth angle: " << std::setprecision(2)
                << max_azimuth_angle << " degrees (total points: " 
                << total_azimuth_points << ", bins: " << total_azimuth_bins << ")" << std::endl;
    }

    polar_plotter->AddSeries(
        "$0", "$1", pangolin::DrawingModeLine, cam_colors[c],
        "mean error(pix) vs polar angle(deg) for cam" + std::to_string(c),
        polar_data_log[c].get());

    azimuth_plotter->AddSeries(
        "$0", "$1", pangolin::DrawingModeLine, cam_colors[c],
        "mean error(pix) vs azimuth angle(deg) for cam" + std::to_string(c),
        azimuth_data_log[c].get());
  }
}

void CamCalib::detectCorners() {
  if (processing_thread) {
    processing_thread->join();
    processing_thread.reset();
  }

  processing_thread.reset(new std::thread([this]() {
    std::cout << "Started detecting corners" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    CalibHelper::detectCorners(this->vio_dataset, this->april_grid,
                               this->calib_corners,
                               this->calib_corners_rejected);

    auto t_end = std::chrono::high_resolution_clock::now();
    auto detect_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_end - t_start)
                         .count();

    std::string path =
        cache_path + cache_dataset_name + "_detected_corners.cereal";
    std::ofstream os(path, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);

    archive(this->calib_corners);
    archive(this->calib_corners_rejected);

    // 统计角点数量（参与/被拒绝）
    size_t good_obs = calib_corners.size();
    size_t bad_obs = calib_corners_rejected.size();
    size_t good_pts = 0;
    size_t bad_pts = 0;
    std::unordered_set<int64_t> frames_with_good;
    std::unordered_set<int64_t> frames_with_rejected;
    for (const auto &kv : calib_corners) {
      good_pts += kv.second.corners.size();
      if (!kv.second.corners.empty())
        frames_with_good.insert(kv.first.frame_id);
    }
    for (const auto &kv : calib_corners_rejected) {
      bad_pts += kv.second.corners.size();
      if (!kv.second.corners.empty())
        frames_with_rejected.insert(kv.first.frame_id);
    }
    std::unordered_set<int64_t> frames_with_any = frames_with_good;
    frames_with_any.insert(frames_with_rejected.begin(),
                           frames_with_rejected.end());

    std::cout << "Done detecting corners. Saved them here: " << path
              << " | good obs: " << good_obs << " good pts: " << good_pts
              << " good images (has good corners): " << frames_with_good.size()
              << " | rejected obs: " << bad_obs << " rejected pts: " << bad_pts
              << " rejected images (has rejected corners): "
              << frames_with_rejected.size()
              << " | images with any corners: " << frames_with_any.size()
              << " | detect time: " << detect_ms << " ms"
              << std::endl;

    // 打印每张图片的角点统计
    // 注意：使用实际从数据集获取的时间戳，而不是从TimeCamId中获取
    std::cout << "\n[PER-IMAGE CORNER STATS] Image corner statistics:" << std::endl;
    std::map<int64_t, std::map<size_t, std::pair<size_t, size_t>>> image_stats;
    for (const auto &kv : calib_corners) {
      image_stats[kv.first.frame_id][kv.first.cam_id].first = kv.second.corners.size();
    }
    for (const auto &kv : calib_corners_rejected) {
      image_stats[kv.first.frame_id][kv.first.cam_id].second = kv.second.corners.size();
    }
    
    // 验证时间戳：从实际图像消息中获取时间戳进行对比
    for (const auto &frame_kv : image_stats) {
      int64_t timestamp_ns = frame_kv.first;
      
      for (const auto &cam_kv : frame_kv.second) {
        size_t cam_id = cam_kv.first;
        size_t good_count = cam_kv.second.first;
        size_t rejected_count = cam_kv.second.second;
        
        // 如果可能，从实际图像消息头获取时间戳
        int64_t actual_timestamp_ns = this->vio_dataset->get_actual_image_timestamp(timestamp_ns, cam_id);
        
        // 标记未使用的变量以避免编译警告
        (void)good_count;
        (void)rejected_count;
        (void)actual_timestamp_ns;
        (void)cam_id;
        
        // std::cout << "  Image timestamp_ns: " << timestamp_ns;
        // if (actual_timestamp_ns != timestamp_ns) {
        //   std::cout << " (actual from header.stamp: " << actual_timestamp_ns << ")";
        // }
        // std::cout << " cam_id: " << cam_id
        //           << " good_corners: " << good_count
        //           << " rejected_corners: " << rejected_count
        //           << std::endl;
      }
    }
  }));

  if (!show_gui) {
    processing_thread->join();
    processing_thread.reset();
  }
}

/// @brief 初始化相机内参
///
/// 该函数用于初始化所有相机的基本内参（焦距和主点）。初始化过程分为两个阶段：
/// 1. 首先尝试使用指定相机模型（如 kb4、ds 等）进行初始化
/// 2. 对于初始化失败的相机，使用理想针孔模型作为备选方案
///
/// 初始化方法基于 Zhang 的标定方法，通过单应性矩阵估计初始内参。
/// 初始化成功后，会设置相机分辨率并输出每个相机的内参值。
///
/// @note 调用此函数前必须先调用 detect_corners 完成角点检测
/// @note 如果数据集图像数量超过 100 张，会跳过部分图像以加快处理速度
void CamCalib::initCamIntrinsics() {
  // 检查是否已检测到角点
  if (calib_corners.empty()) {
    std::cerr << "No corners detected. Press detect_corners to start corner "
                 "detection."
              << std::endl;
    return;
  }

  std::cout << "Started camera intrinsics initialization" << std::endl;

  // 如果优化器未初始化，创建新的优化器对象
  if (!calib_opt) calib_opt.reset(new PosesOptimization);

  // 根据相机数量和类型重置标定参数
  calib_opt->resetCalib(vio_dataset->get_num_cams(), cam_types);

  // 标记每个相机是否已成功初始化
  std::vector<bool> cam_initialized(vio_dataset->get_num_cams(), false);

  // 设置图像采样间隔：如果图像数量超过 100 张，每隔 3 张采样一次以加快速度
  int inc = 1;
  if (vio_dataset->get_image_timestamps().size() > 100) inc = 3;

  // 第一阶段：使用指定相机模型初始化内参
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    // 遍历图像序列，寻找能够成功初始化的图像
    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size();
         i += inc) {
      const int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
      const std::vector<basalt::ImageData> &img_vec =
          vio_dataset->get_image_data(timestamp_ns);

      TimeCamId tcid(timestamp_ns, j);

      // 检查该时间戳和相机 ID 是否检测到角点
      if (calib_corners.find(tcid) != calib_corners.end()) {
        CalibCornerData cid = calib_corners.at(tcid);

        // 存储初始内参（4 个参数：fx, fy, cx, cy）
        Eigen::Vector4d init_intr;

        // 使用 Zhang 方法通过单应性矩阵估计初始内参
        // 该方法适用于各种相机模型（kb4、ds、eucm 等）
        bool success = CalibHelper::initializeIntrinsics(
            cid.corners, cid.corner_ids, april_grid, img_vec[j].img->w,
            img_vec[j].img->h, init_intr);

        if (success) {
          cam_initialized[j] = true;
          // 将初始内参设置到相机模型中（会根据相机类型转换为相应参数）
          calib_opt->calib->intrinsics[j].setFromInit(init_intr);
          break;  // 成功初始化后跳出循环
        }
      }
    }
  }

  // 第二阶段：对于初始化失败的相机，使用理想针孔模型作为备选方案
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    if (!cam_initialized[j]) {
      // 收集该相机所有包含足够角点的图像数据
      std::vector<CalibCornerData *> pinhole_corners;
      int w = 0;  // 图像宽度
      int h = 0;  // 图像高度

      for (size_t i = 0; i < vio_dataset->get_image_timestamps().size();
           i += inc) {
        const int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];
        const std::vector<basalt::ImageData> &img_vec =
            vio_dataset->get_image_data(timestamp_ns);

        TimeCamId tcid(timestamp_ns, j);

        auto it = calib_corners.find(tcid);
        if (it != calib_corners.end()) {
          // 只使用角点数量大于 8 的图像（确保有足够的约束）
          if (it->second.corners.size() > 8) {
            pinhole_corners.emplace_back(&it->second);
          }
        }

        // 记录图像尺寸
        w = img_vec[j].img->w;
        h = img_vec[j].img->h;
      }

      BASALT_ASSERT(w > 0 && h > 0);

      Eigen::Vector4d init_intr;

      // 使用理想针孔模型初始化内参（基于多张图像的平均值）
      bool success = CalibHelper::initializeIntrinsicsPinhole(
          pinhole_corners, april_grid, w, h, init_intr);

      if (success) {
        cam_initialized[j] = true;

        std::cout << "Initialized camera " << j
                  << " with pinhole model. You should set pinhole model for "
                     "this camera!"
                  << std::endl;
        calib_opt->calib->intrinsics[j].setFromInit(init_intr);
      }
    }
  }

  // 输出所有相机的初始化结果
  std::cout << "Done camera intrinsics initialization:" << std::endl;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    std::cout << "Cam " << j << ": "
              << calib_opt->calib->intrinsics[j].getParam().transpose()
              << std::endl;
  }

  // 设置相机分辨率
  {
    size_t img_idx = 1;
    int64_t t_ns = vio_dataset->get_image_timestamps()[img_idx];
    auto img_data = vio_dataset->get_image_data(t_ns);

    // 查找包含所有有效图像的帧
    while (img_idx < vio_dataset->get_image_timestamps().size()) {
      bool img_data_valid = true;
      for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
        if (!img_data[i].img.get()) img_data_valid = false;
      }

      if (!img_data_valid) {
        // 如果当前帧无效，继续查找下一帧
        img_idx++;
        int64_t t_ns_new = vio_dataset->get_image_timestamps()[img_idx];
        img_data = vio_dataset->get_image_data(t_ns_new);
      } else {
        break;  // 找到有效帧后退出
      }
    }

    // 提取所有相机的分辨率
    Eigen::aligned_vector<Eigen::Vector2i> res;

    for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
      res.emplace_back(img_data[i].img->w, img_data[i].img->h);
    }

    // 将分辨率设置到优化器中
    calib_opt->setResolution(res);
  }
}

void CamCalib::initCamPoses() {
  if (calib_corners.empty()) {
    std::cerr << "No corners detected. Press detect_corners to start corner "
                 "detection."
              << std::endl;
    return;
  }

  if (!calib_opt.get() || !calib_opt->calibInitialized()) {
    std::cerr << "No initial intrinsics. Press init_intrinsics initialize "
                 "intrinsics"
              << std::endl;
    return;
  }

  if (processing_thread) {
    processing_thread->join();
    processing_thread.reset();
  }

  std::cout << "Started initial camera pose computation " << std::endl;

  CalibHelper::initCamPoses(calib_opt->calib,
                            april_grid.aprilgrid_corner_pos_3d,
                            this->calib_corners, this->calib_init_poses);

  std::string path = cache_path + cache_dataset_name + "_init_poses.cereal";
  std::ofstream os(path, std::ios::binary);
  cereal::BinaryOutputArchive archive(os);

  archive(this->calib_init_poses);

  std::cout << "Done initial camera pose computation. Saved them here: " << path
            << std::endl;
}

void CamCalib::initCamExtrinsics() {
  if (calib_init_poses.empty()) {
    std::cerr << "No initial camera poses. Press init_cam_poses initialize "
                 "camera poses "
              << std::endl;
    return;
  }

  if (!calib_opt.get() || !calib_opt->calibInitialized()) {
    std::cerr << "No initial intrinsics. Press init_intrinsics initialize "
                 "intrinsics"
              << std::endl;
    return;
  }

  // Camera graph. Stores the edge from i to j with weight w and timestamp. i
  // and j should be sorted;
  std::map<std::pair<size_t, size_t>, std::pair<int, int64_t>> cam_graph;

  // Construct the graph.
  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[i];

    for (size_t cam_i = 0; cam_i < vio_dataset->get_num_cams(); cam_i++) {
      TimeCamId tcid_i(timestamp_ns, cam_i);

      auto it = calib_init_poses.find(tcid_i);
      if (it == calib_init_poses.end() || it->second.num_inliers < MIN_CORNERS)
        continue;

      for (size_t cam_j = cam_i + 1; cam_j < vio_dataset->get_num_cams();
           cam_j++) {
        TimeCamId tcid_j(timestamp_ns, cam_j);

        auto it2 = calib_init_poses.find(tcid_j);
        if (it2 == calib_init_poses.end() ||
            it2->second.num_inliers < MIN_CORNERS)
          continue;

        std::pair<size_t, size_t> edge_id(cam_i, cam_j);

        int curr_weight = cam_graph[edge_id].first;
        int new_weight =
            std::min(it->second.num_inliers, it2->second.num_inliers);

        if (curr_weight < new_weight) {
          cam_graph[edge_id] = std::make_pair(new_weight, timestamp_ns);
        }
      }
    }
  }

  std::vector<bool> cameras_initialized(vio_dataset->get_num_cams(), false);
  cameras_initialized[0] = true;
  size_t last_camera = 0;
  calib_opt->calib->T_i_c[0] = Sophus::SE3d();  // Identity

  auto next_max_weight_edge = [&](size_t cam_id) {
    int max_weight = -1;
    std::pair<int, int64_t> res(-1, -1);

    for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
      if (cameras_initialized[i]) continue;

      std::pair<size_t, size_t> edge_id;

      if (i < cam_id) {
        edge_id = std::make_pair(i, cam_id);
      } else if (i > cam_id) {
        edge_id = std::make_pair(cam_id, i);
      }

      auto it = cam_graph.find(edge_id);
      if (it != cam_graph.end() && max_weight < it->second.first) {
        max_weight = it->second.first;
        res.first = i;
        res.second = it->second.second;
      }
    }

    return res;
  };

  for (size_t i = 0; i < vio_dataset->get_num_cams() - 1; i++) {
    std::pair<int, int64_t> res = next_max_weight_edge(last_camera);

    std::cout << "Initializing camera pair " << last_camera << " " << res.first
              << std::endl;

    if (res.first >= 0) {
      size_t new_camera = res.first;

      TimeCamId tcid_last(res.second, last_camera);
      TimeCamId tcid_new(res.second, new_camera);

      calib_opt->calib->T_i_c[new_camera] =
          calib_opt->calib->T_i_c[last_camera] *
          calib_init_poses.at(tcid_last).T_a_c.inverse() *
          calib_init_poses.at(tcid_new).T_a_c;

      last_camera = new_camera;
      cameras_initialized[last_camera] = true;
    }
  }

  std::cout << "Done camera extrinsics initialization:" << std::endl;
  for (size_t j = 0; j < vio_dataset->get_num_cams(); j++) {
    std::cout << "T_c0_c" << j << ":\n"
              << calib_opt->calib->T_i_c[j].matrix() << std::endl;
  }
}  // namespace basalt

void CamCalib::initOptimization() {
  if (!calib_opt) {
    std::cerr << "Calibration is not initialized. Initialize calibration first!"
              << std::endl;
    return;
  }

  if (calib_init_poses.empty()) {
    std::cerr << "No initial camera poses. Press init_cam_poses initialize "
                 "camera poses "
              << std::endl;
    return;
  }

  calib_opt->setAprilgridCorners3d(april_grid.aprilgrid_corner_pos_3d);

  std::unordered_set<TimeCamId> invalid_frames;
  size_t filtered_by_corner_count = 0;
  size_t filtered_points_by_corner_count = 0;
  
  // 第一轮过滤：角点数量不足的帧
  for (const auto &kv : calib_corners) {
    if (kv.second.corner_ids.size() < MIN_CORNERS) {
      invalid_frames.insert(kv.first);
      filtered_by_corner_count++;
      filtered_points_by_corner_count += kv.second.corners.size();
    }
  }

  size_t filtered_by_pose_init = 0;
  size_t filtered_points_by_pose_init = 0;
  
  for (size_t j = 0; j < vio_dataset->get_image_timestamps().size(); ++j) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[j];

    int max_inliers = -1;
    int max_inliers_idx = -1;

    for (size_t cam_id = 0; cam_id < calib_opt->calib->T_i_c.size(); cam_id++) {
      TimeCamId tcid(timestamp_ns, cam_id);
      const auto cp_it = calib_init_poses.find(tcid);
      if (cp_it != calib_init_poses.end()) {
        if ((int)cp_it->second.num_inliers > max_inliers) {
          max_inliers = cp_it->second.num_inliers;
          max_inliers_idx = cam_id;
        }
      }
    }

    if (max_inliers >= (int)MIN_CORNERS) {
      TimeCamId tcid(timestamp_ns, max_inliers_idx);
      const auto cp_it = calib_init_poses.find(tcid);

      // Initial pose
      calib_opt->addPoseMeasurement(
          timestamp_ns, cp_it->second.T_a_c *
                            calib_opt->calib->T_i_c[max_inliers_idx].inverse());
    } else {
      // Set all frames invalid if we do not have initial pose
      for (size_t cam_id = 0; cam_id < calib_opt->calib->T_i_c.size();
           cam_id++) {
        TimeCamId tcid(timestamp_ns, cam_id);
        if (invalid_frames.count(tcid) == 0) {
          invalid_frames.emplace(tcid);
          filtered_by_pose_init++;
          // 计算该帧被过滤掉的点数
          const auto corners_it = calib_corners.find(tcid);
          if (corners_it != calib_corners.end()) {
            filtered_points_by_pose_init += corners_it->second.corners.size();
          }
        }
      }
    }
  }

  size_t total_points_added = 0;
  size_t total_detected_points = 0;
  for (const auto &kv : calib_corners) {
    total_detected_points += kv.second.corners.size();
    if (invalid_frames.count(kv.first) == 0) {
      calib_opt->addAprilgridMeasurement(kv.first.frame_id, kv.first.cam_id,
                                         kv.second.corners,
                                         kv.second.corner_ids);
      total_points_added += kv.second.corners.size();
    }
  }

  calib_opt->init();
  computeProjections();

  std::cout << "Initialized optimization. Total points added: " << total_points_added 
            << ", invalid frames: " << invalid_frames.size()
            << ", poses added: " << calib_opt->getTimestampToPose().size() << std::endl;
  std::cout << "[FILTERING STATS] Detected points: " << total_detected_points
            << ", Added points: " << total_points_added
            << ", Filtered points: " << (total_detected_points - total_points_added) << std::endl;
  std::cout << "[FILTERING BREAKDOWN] Filtered by corner count < " << MIN_CORNERS 
            << ": " << filtered_by_corner_count << " frames, " 
            << filtered_points_by_corner_count << " points" << std::endl;
  std::cout << "[FILTERING BREAKDOWN] Filtered by pose init inliers < " << MIN_CORNERS 
            << ": " << filtered_by_pose_init << " frames, " 
            << filtered_points_by_pose_init << " points" << std::endl;
  
  // Check if we have valid poses and points
  if (calib_opt->getTimestampToPose().empty()) {
    std::cerr << "[ERROR] No valid poses added to optimization. "
              << "This may be because all frames have fewer than " << MIN_CORNERS 
              << " inliers." << std::endl;
  }
  
  if (total_points_added == 0) {
    std::cerr << "[ERROR] No valid points added to optimization. "
              << "This may be because all frames are marked as invalid." << std::endl;
  }
}  // namespace basalt

void CamCalib::loadDataset() {
  basalt::DatasetIoInterfacePtr dataset_io =
      basalt::DatasetIoFactory::getDatasetIo(dataset_type);

  dataset_io->read(dataset_path);

  vio_dataset = dataset_io->get_data();
  setNumCameras(vio_dataset->get_num_cams());

  if (skip_images > 1) {
    std::vector<int64_t> new_image_timestamps;
    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
      if (i % skip_images == 0)
        new_image_timestamps.push_back(vio_dataset->get_image_timestamps()[i]);
    }

    vio_dataset->get_image_timestamps() = new_image_timestamps;
  }

  // load detected corners if they exist
  {
    std::string path =
        cache_path + cache_dataset_name + "_detected_corners.cereal";

    std::ifstream is(path, std::ios::binary);

    if (is.good()) {
      cereal::BinaryInputArchive archive(is);

      calib_corners.clear();
      calib_corners_rejected.clear();
      archive(calib_corners);
      archive(calib_corners_rejected);

      std::cout << "Loaded detected corners from: " << path << std::endl;
    } else {
      std::cout << "No pre-processed detected corners found" << std::endl;
    }
  }

  // load initial poses if they exist
  {
    std::string path = cache_path + cache_dataset_name + "_init_poses.cereal";

    std::ifstream is(path, std::ios::binary);

    if (is.good()) {
      cereal::BinaryInputArchive archive(is);

      calib_init_poses.clear();
      archive(calib_init_poses);

      std::cout << "Loaded initial poses from: " << path << std::endl;
    } else {
      std::cout << "No pre-processed initial poses found" << std::endl;
    }
  }

  // load calibration if exist
  {
    if (!calib_opt) calib_opt.reset(new PosesOptimization);

    calib_opt->loadCalib(cache_path);
  }

  reprojected_corners.clear();
  reprojected_vignette.clear();

  if (show_gui) {
    show_frame = 0;

    show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
    show_frame.Meta().gui_changed = true;
  }
}

void CamCalib::optimize() { optimizeWithParam(true); }

bool CamCalib::optimizeWithParam(bool print_info,
                                 std::map<std::string, double> *stats) {
  if (calib_init_poses.empty()) {
    std::cerr << "No initial camera poses. Press init_cam_poses initialize "
                 "camera poses "
              << std::endl;
    return true;
  }

  if (!calib_opt.get() || !calib_opt->calibInitialized()) {
    std::cerr << "No initial intrinsics. Press init_intrinsics initialize "
                 "intrinsics"
              << std::endl;
    return true;
  }

  bool converged = true;

  if (calib_opt) {
    // calib_opt->compute_projections();
    double error;
    double reprojection_error;
    int num_points;

    // 计算优化前的每张图片统计
    if (print_info) {
      std::map<std::pair<int64_t, size_t>, std::vector<double>> per_image_reproj_before;
      std::map<std::pair<int64_t, size_t>, double> per_image_tz_before;
      
      for (const auto& acd : calib_opt->getAprilgridCornersMeasurements()) {
        auto it_pose = calib_opt->getTimestampToPose().find(acd.timestamp_ns);
        if (it_pose == calib_opt->getTimestampToPose().end()) continue;

        Sophus::SE3d T_w_i = it_pose->second;
        Sophus::SE3d T_w_c =
            T_w_i * calib_opt->calib->T_i_c[acd.cam_id];
        Sophus::SE3d T_c_w = T_w_c.inverse();
        Eigen::Matrix4d T_c_w_m = T_c_w.matrix();
        
        std::pair<int64_t, size_t> key = {acd.timestamp_ns, acd.cam_id};
        per_image_tz_before[key] = T_w_c.translation()[2];

        std::visit(
            [&](const auto& cam) {
              for (size_t k = 0; k < acd.corner_pos.size(); k++) {
                Eigen::Vector4d point3d =
                    T_c_w_m * april_grid.aprilgrid_corner_pos_3d[acd.corner_id[k]];
                Eigen::Vector2d proj;
                if (cam.project(point3d, proj)) {
                  double e = (acd.corner_pos[k] - proj).norm();
                  per_image_reproj_before[key].push_back(e);
                }
              }
            },
            calib_opt->calib->intrinsics[acd.cam_id].variant);
      }

      // 打印优化前的统计
      std::cout << "\n[BEFORE OPTIMIZATION] Per-image reprojection error and pose t_z:" << std::endl;
      
      // 打印优化前 t_z < 0 的图像
      for (const auto& kv : per_image_tz_before) {
        const auto& key = kv.first;
        double tz = kv.second;
        if (tz < 0) {
          std::cout << "\n[BEFORE OPTIMIZATION] Images with t_z < 0:" << std::endl;
          int64_t timestamp_ns = key.first;
          size_t cam_id = key.second;
          // 尝试获取图像文件名或使用时间戳
          std::string image_name = "timestamp_" + std::to_string(timestamp_ns);
          // 尝试从数据集获取实际时间戳（如果可用）
          if (vio_dataset) {
            int64_t actual_timestamp = vio_dataset->get_actual_image_timestamp(timestamp_ns, cam_id);
            if (actual_timestamp != timestamp_ns) {
              image_name += "_actual_" + std::to_string(actual_timestamp);
            }
          }
          std::cout << "  Image: " << image_name 
                    << " cam_id: " << cam_id
                    << " t_z: " << tz << std::endl;
        }
      }
      
      for (const auto& kv : per_image_reproj_before) {
        const auto& key = kv.first;
        const auto& errors = kv.second;
        if (errors.empty()) continue;
        
        double sum = 0.0, sum_sq = 0.0, max_err = 0.0;
        for (double e : errors) {
          sum += e;
          sum_sq += e * e;
          max_err = std::max(max_err, e);
        }
        double mean = sum / errors.size();
        double std_dev = std::sqrt((sum_sq / errors.size()) - (mean * mean));
        double tz = per_image_tz_before[key];
        
        // 标记未使用的变量以避免编译警告
        (void)std_dev;
        (void)tz;
        
        // std::cout << "  Image timestamp_ns: " << key.first 
        //           << " cam_id: " << key.second
        //           << " reproj_mean: " << mean
        //           << " reproj_std: " << std_dev
        //           << " reproj_max: " << max_err
        //           << " t_z: " << tz
        //           << std::endl;
      }
    }

    auto start = std::chrono::high_resolution_clock::now();

    converged = calib_opt->optimize(opt_intr, huber_thresh, stop_thresh, error,
                                    num_points, reprojection_error);

    auto finish = std::chrono::high_resolution_clock::now();

    if (stats) {
      stats->clear();

      stats->emplace("energy_error", error);
      stats->emplace("num_points", num_points);
      stats->emplace("mean_energy_error", error / num_points);
      stats->emplace("reprojection_error", reprojection_error);
      stats->emplace("mean_reprojection_error",
                     reprojection_error / num_points);
    }

    if (print_info) {
      // 仅统计参与优化的帧/点（使用与优化相同的数据集）
      double reproj_max = 0.0;
      double reproj_sum = 0.0;
      double reproj_sum_sq = 0.0;
      int reproj_count = 0;

      for (const auto& acd : calib_opt->getAprilgridCornersMeasurements()) {
        auto it_pose = calib_opt->getTimestampToPose().find(acd.timestamp_ns);
        if (it_pose == calib_opt->getTimestampToPose().end()) continue;

        Sophus::SE3d T_w_i = it_pose->second;
        Sophus::SE3d T_w_c =
            T_w_i * calib_opt->calib->T_i_c[acd.cam_id];
        Sophus::SE3d T_c_w = T_w_c.inverse();
        Eigen::Matrix4d T_c_w_m = T_c_w.matrix();

        std::visit(
            [&](const auto& cam) {
              for (size_t k = 0; k < acd.corner_pos.size(); k++) {
                Eigen::Vector4d point3d =
                    T_c_w_m * april_grid.aprilgrid_corner_pos_3d[acd.corner_id[k]];
                Eigen::Vector2d proj;
                if (cam.project(point3d, proj)) {
                  double e = (acd.corner_pos[k] - proj).norm();
                  reproj_max = std::max(reproj_max, e);
                  reproj_sum += e;
                  reproj_sum_sq += e * e;
                  reproj_count++;
                }
              }
            },
            calib_opt->calib->intrinsics[acd.cam_id].variant);
      }

      double reproj_mean = reproj_count > 0 ? reproj_sum / reproj_count : 0.0;
      double reproj_std = reproj_count > 0
                              ? std::sqrt((reproj_sum_sq / reproj_count) -
                                          (reproj_mean * reproj_mean))
                              : 0.0;

      std::cout << "==================================" << std::endl;

      for (size_t i = 0; i < vio_dataset->get_num_cams(); i++) {
        std::cout << "intrinsics " << i << ": "
                  << calib_opt->calib->intrinsics[i].getParam().transpose()
                  << std::endl;
        std::cout << "T_i_c" << i << ":\n"
                  << calib_opt->calib->T_i_c[i].matrix() << std::endl;
      }

      std::cout << "Current error: " << error << " num_points " << num_points
                << " mean_error " << error / num_points
                << " reprojection_error " << reprojection_error
                << " mean reprojection " << reprojection_error / num_points
                << " reproj_std " << reproj_std
                << " reproj_max " << reproj_max
                << " opt_time "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       finish - start)
                       .count()
                << "ms." << std::endl;

      // 计算优化后的每张图片统计
      std::map<std::pair<int64_t, size_t>, std::vector<double>> per_image_reproj_after;
      std::map<std::pair<int64_t, size_t>, double> per_image_tz_after;
      
      for (const auto& acd : calib_opt->getAprilgridCornersMeasurements()) {
        auto it_pose = calib_opt->getTimestampToPose().find(acd.timestamp_ns);
        if (it_pose == calib_opt->getTimestampToPose().end()) continue;

        Sophus::SE3d T_w_i = it_pose->second;
        Sophus::SE3d T_w_c =
            T_w_i * calib_opt->calib->T_i_c[acd.cam_id];
        Sophus::SE3d T_c_w = T_w_c.inverse();
        Eigen::Matrix4d T_c_w_m = T_c_w.matrix();
        
        std::pair<int64_t, size_t> key = {acd.timestamp_ns, acd.cam_id};
        per_image_tz_after[key] = T_w_c.translation()[2];

        std::visit(
            [&](const auto& cam) {
              for (size_t k = 0; k < acd.corner_pos.size(); k++) {
                Eigen::Vector4d point3d =
                    T_c_w_m * april_grid.aprilgrid_corner_pos_3d[acd.corner_id[k]];
                Eigen::Vector2d proj;
                if (cam.project(point3d, proj)) {
                  double e = (acd.corner_pos[k] - proj).norm();
                  per_image_reproj_after[key].push_back(e);
                }
              }
            },
            calib_opt->calib->intrinsics[acd.cam_id].variant);
      }

      // 打印优化后的统计
      std::cout << "\n[AFTER OPTIMIZATION] Per-image reprojection error and pose t_z:" << std::endl;
      
      // 打印优化后 t_z < 0 的图像
      for (const auto& kv : per_image_tz_after) {
        const auto& key = kv.first;
        double tz = kv.second;
        if (tz < 0) {
          std::cout << "\n[AFTER OPTIMIZATION] Images with t_z < 0:" << std::endl;
          int64_t timestamp_ns = key.first;
          size_t cam_id = key.second;
          // 尝试获取图像文件名或使用时间戳
          std::string image_name = "timestamp_" + std::to_string(timestamp_ns);
          // 尝试从数据集获取实际时间戳（如果可用）
          if (vio_dataset) {
            int64_t actual_timestamp = vio_dataset->get_actual_image_timestamp(timestamp_ns, cam_id);
            if (actual_timestamp != timestamp_ns) {
              image_name += "_actual_" + std::to_string(actual_timestamp);
            }
          }
          std::cout << "  Image: " << image_name 
                    << " cam_id: " << cam_id
                    << " t_z: " << tz << std::endl;
        }
      }
      
      for (const auto& kv : per_image_reproj_after) {
        const auto& key = kv.first;
        const auto& errors = kv.second;
        if (errors.empty()) continue;
        
        double sum = 0.0, sum_sq = 0.0, max_err = 0.0;
        for (double e : errors) {
          sum += e;
          sum_sq += e * e;
          max_err = std::max(max_err, e);
        }
        double mean = sum / errors.size();
        double std_dev = std::sqrt((sum_sq / errors.size()) - (mean * mean));
        double tz = per_image_tz_after[key];
        
        // 标记未使用的变量以避免编译警告
        (void)std_dev;
        (void)tz;
        
        // std::cout << "  Image timestamp_ns: " << key.first 
        //           << " cam_id: " << key.second
        //           << " reproj_mean: " << mean
        //           << " reproj_std: " << std_dev
        //           << " reproj_max: " << max_err
        //           << " t_z: " << tz
        //           << std::endl;
      }

      if (converged) std::cout << "Optimization Converged !!" << std::endl;

      std::cout << "==================================" << std::endl;
    }

    if (show_gui) {
      computeProjections();
    }
  }

  return converged;
}

void CamCalib::saveCalib() {
  if (calib_opt) {
    calib_opt->saveCalib(cache_path);

    std::cout << "Saved calibration in " << cache_path << "calibration.json"
              << std::endl;
  }
}

// 辅助函数：绘制十字架
static void drawCross(cv::Mat& img, const cv::Point2d& pt, int size, 
                      const cv::Scalar& color, int thickness) {
  // 水平线
  cv::line(img, 
           cv::Point2d(pt.x - size, pt.y),
           cv::Point2d(pt.x + size, pt.y),
           color, thickness);
  // 垂直线
  cv::line(img,
           cv::Point2d(pt.x, pt.y - size),
           cv::Point2d(pt.x, pt.y + size),
           color, thickness);
}

void CamCalib::saveCornerImages() {
  if (!vio_dataset.get()) {
    std::cerr << "No dataset loaded. Please load dataset first." << std::endl;
    return;
  }

  // 获取当前工作目录
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) == nullptr) {
    std::cerr << "Failed to get current working directory." << std::endl;
    return;
  }
  std::string work_dir = std::string(cwd) + "/";

  // 创建保存目录 cam0 或 cam1
  for (size_t cam_id = 0; cam_id < vio_dataset->get_num_cams(); cam_id++) {
    std::string cam_dir = work_dir + "cam" + std::to_string(cam_id) + "/";
    if (!fs::exists(cam_dir)) {
      fs::create_directory(cam_dir);
    }
  }

  std::cout << "Started saving corner images for all frames..." << std::endl;

  // 遍历所有帧
  size_t total_saved = 0;
  for (size_t frame_id = 0; frame_id < vio_dataset->get_image_timestamps().size(); frame_id++) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[frame_id];
    const std::vector<ImageData> &img_vec =
        vio_dataset->get_image_data(timestamp_ns);

    for (size_t cam_id = 0; cam_id < vio_dataset->get_num_cams(); cam_id++) {
      if (!img_vec[cam_id].img.get()) continue;

      TimeCamId tcid(timestamp_ns, cam_id);

      // 将图像转换为OpenCV Mat
      const auto &img = img_vec[cam_id].img;
      cv::Mat cv_img(img->h, img->w, CV_16UC1, (void *)img->ptr);
      cv::Mat cv_img_8u;
      cv_img.convertTo(cv_img_8u, CV_8U, 1.0 / 256.0);
      cv::Mat cv_img_color;
      cv::cvtColor(cv_img_8u, cv_img_color, cv::COLOR_GRAY2BGR);

      // 在同一张图片上绘制所有三种角点
      cv::Mat img_all = cv_img_color.clone();

      // 1. 绘制检测到的角点（红色十字架，长度6，宽度1）
      if (calib_corners.find(tcid) != calib_corners.end()) {
        const CalibCornerData &cr = calib_corners.at(tcid);
        for (size_t i = 0; i < cr.corners.size(); i++) {
          cv::Point2d pt(cr.corners[i].x(), cr.corners[i].y());
          drawCross(img_all, pt, 3, cv::Scalar(0, 0, 255), 1);  // BGR格式，红色
        }
      }

      // 2. 绘制被拒绝的角点（蓝色十字架，长度6，宽度1）
      if (calib_corners_rejected.find(tcid) != calib_corners_rejected.end()) {
        const CalibCornerData &cr_rej = calib_corners_rejected.at(tcid);
        for (size_t i = 0; i < cr_rej.corners.size(); i++) {
          cv::Point2d pt(cr_rej.corners[i].x(), cr_rej.corners[i].y());
          drawCross(img_all, pt, 2, cv::Scalar(255, 0, 0), 1);  // BGR格式，蓝色
        }
      }

      // 3. 绘制优化后的投影角点（绿色十字架，长度4，宽度1）
      // 同时计算重投影误差
      double reproj_mean = 0.0;
      double reproj_max = 0.0;
      int reproj_count = 0;
      
      if (reprojected_corners.find(tcid) != reprojected_corners.end()) {
        if (calib_corners.count(tcid) > 0 &&
            calib_corners.at(tcid).corner_ids.size() >= MIN_CORNERS) {
          const auto &rc = reprojected_corners.at(tcid);
          const CalibCornerData &cr = calib_corners.at(tcid);
          
          // 计算重投影误差：检测角点与投影角点的差异
          for (size_t i = 0; i < rc.corners_proj.size(); i++) {
            if (!rc.corners_proj_success[i]) continue;
            
            // 查找对应的检测角点（通过角点ID匹配）
            for (size_t j = 0; j < cr.corner_ids.size(); j++) {
              if (cr.corner_ids[j] == static_cast<int>(i)) {
                double e = (cr.corners[j] - rc.corners_proj[i]).norm();
                reproj_mean += e;
                reproj_max = std::max(reproj_max, e);
                reproj_count++;
                break;
              }
            }
          }
          
          // 绘制投影角点
          for (size_t i = 0; i < rc.corners_proj.size(); i++) {
            if (!rc.corners_proj_success[i]) continue;
            cv::Point2d pt(rc.corners_proj[i].x(), rc.corners_proj[i].y());
            drawCross(img_all, pt, 4, cv::Scalar(0, 255, 0), 1);  // BGR格式，绿色
          }
        }
      }
      
      // 计算平均重投影误差
      if (reproj_count > 0) {
        reproj_mean /= reproj_count;
      }

      // 在图片左下角绘制重投影误差信息
      if (reproj_count > 0) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "Mean: " << reproj_mean << " Max: " << reproj_max;
        std::string text = ss.str();
        
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.6;
        int thickness = 2;
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        
        // 在左下角绘制，留出一些边距
        int margin = 10;
        cv::Point text_pos(margin, img_all.rows - margin);
        
        // 绘制文字背景（黑色半透明）
        cv::rectangle(img_all, 
                     cv::Point(text_pos.x - 5, text_pos.y - text_size.height - 5),
                     cv::Point(text_pos.x + text_size.width + 5, text_pos.y + 5),
                     cv::Scalar(0, 0, 0), -1);
        
        // 绘制白色文字
        cv::putText(img_all, text, text_pos, font_face, font_scale, 
                   cv::Scalar(255, 255, 255), thickness);
      }

      // 使用时间戳作为文件名（与输入图片一致）
      std::string cam_dir = work_dir + "cam" + std::to_string(cam_id) + "/";
      std::string filename = cam_dir + std::to_string(timestamp_ns) + ".png";
      cv::imwrite(filename, img_all);
      total_saved++;
    }

    // 每处理100帧输出一次进度
    if ((frame_id + 1) % 100 == 0) {
      std::cout << "Processed " << (frame_id + 1) << " / " 
                << vio_dataset->get_image_timestamps().size() << " frames" << std::endl;
    }
  }

  std::cout << "Done saving corner images. Total saved: " << total_saved 
            << " images to " << work_dir << "cam*/" << std::endl;
}

void CamCalib::drawImageOverlay(pangolin::View &v, size_t cam_id) {
  UNUSED(v);

  size_t frame_id = show_frame;

  if (vio_dataset && frame_id < vio_dataset->get_image_timestamps().size()) {
    int64_t timestamp_ns = vio_dataset->get_image_timestamps()[frame_id];
    TimeCamId tcid(timestamp_ns, cam_id);

    if (show_corners) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (calib_corners.find(tcid) != calib_corners.end()) {
        const CalibCornerData &cr = calib_corners.at(tcid);
        const CalibCornerData &cr_rej = calib_corners_rejected.at(tcid);

        for (size_t i = 0; i < cr.corners.size(); i++) {
          // the radius is the threshold used for maximum displacement. The
          // search region is slightly larger.
          const float radius = static_cast<float>(cr.radii[i]);
          const Eigen::Vector2f c = cr.corners[i].cast<float>();
          pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

          if (show_ids)
            pangolin::default_font().Text("%d", cr.corner_ids[i]).Draw(c[0], c[1]);
        }

        pangolin::default_font()
            .Text("Detected %d corners (%d rejected)", cr.corners.size(),
                  cr_rej.corners.size())
            .Draw(5, 50);

        if (show_corners_rejected) {
          glColor3f(1.0, 0.5, 0.0);

          for (size_t i = 0; i < cr_rej.corners.size(); i++) {
            // the radius is the threshold used for maximum displacement. The
            // search region is slightly larger.
            const float radius = static_cast<float>(cr_rej.radii[i]);
            const Eigen::Vector2f c = cr_rej.corners[i].cast<float>();
            pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

            if (show_ids)
              pangolin::default_font()
                  .Text("%d", cr_rej.corner_ids[i])
                  .Draw(c[0], c[1]);
          }
        }

      } else {
        glLineWidth(1.0);

        pangolin::default_font().Text("Corners not processed").Draw(5, 50);
      }
    }

    if (show_init_reproj) {
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (calib_init_poses.find(tcid) != calib_init_poses.end()) {
        const CalibInitPoseData &cr = calib_init_poses.at(tcid);

        for (size_t i = 0; i < cr.reprojected_corners.size(); i++) {
          Eigen::Vector2d c = cr.reprojected_corners[i];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          if (show_ids) pangolin::default_font().Text("%d", i).Draw(c[0], c[1]);
        }

        pangolin::default_font()
            .Text("Initial pose with %d inliers", cr.num_inliers)
            .Draw(5, 100);

      } else {
        pangolin::default_font().Text("Initial pose not processed").Draw(5, 100);
      }
    }

    if (show_opt) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 1.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (reprojected_corners.find(tcid) != reprojected_corners.end()) {
        if (calib_corners.count(tcid) > 0 &&
            calib_corners.at(tcid).corner_ids.size() >= MIN_CORNERS) {
          const auto &rc = reprojected_corners.at(tcid);

          for (size_t i = 0; i < rc.corners_proj.size(); i++) {
            if (!rc.corners_proj_success[i]) continue;

            Eigen::Vector2d c = rc.corners_proj[i];
            pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

            if (show_ids) pangolin::default_font().Text("%d", i).Draw(c[0], c[1]);
          }
        } else {
          pangolin::default_font().Text("Too few corners detected.").Draw(5, 150);
        }
      }
    }

    if (show_vign) {
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (reprojected_vignette.find(tcid) != reprojected_vignette.end()) {
        if (calib_corners.count(tcid) > 0 &&
            calib_corners.at(tcid).corner_ids.size() >= MIN_CORNERS) {
          const auto &rc = reprojected_vignette.at(tcid);

          bool has_errors = false;
          auto it = reprojected_vignette_error.find(tcid);
          if (it != reprojected_vignette_error.end()) has_errors = true;

          for (size_t i = 0; i < rc.corners_proj.size(); i++) {
            if (!rc.corners_proj_success[i]) continue;

            Eigen::Vector2d c = rc.corners_proj[i].head<2>();
            pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

            if (show_ids) {
              if (has_errors) {
                pangolin::default_font()
                    .Text("%d(%f)", i, it->second[i])
                    .Draw(c[0], c[1]);
              } else {
                pangolin::default_font().Text("%d", i).Draw(c[0], c[1]);
              }
            }
          }
        } else {
          pangolin::default_font().Text("Too few corners detected.").Draw(5, 200);
        }
      }
    }
  }
}

bool CamCalib::hasCorners() const { return !calib_corners.empty(); }

}  // namespace basalt
