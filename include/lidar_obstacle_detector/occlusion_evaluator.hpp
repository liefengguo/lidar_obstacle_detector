#pragma once

#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace lidar_obstacle_detector
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

enum class OcclusionState : uint8_t
{
  NO_OCCLUSION = 0,
  MILD_OCCLUSION = 1,
  MODERATE_OCCLUSION = 2,
  SEVERE_OCCLUSION = 3,
  DYNAMIC_OCCLUSION = 4
};

struct OcclusionEvaluatorConfig
{
  size_t sector_count = 36;
  size_t min_points_per_sector = 15;
  float mild_ratio = 0.15f;
  float moderate_ratio = 0.4f;
  float severe_ratio = 0.7f;
  float near_distance_threshold = 12.0f;
  double dynamic_time_window = 2.0;
  size_t dynamic_transition_threshold = 3;
};

struct OcclusionEvaluation
{
  OcclusionState state = OcclusionState::NO_OCCLUSION;
  float occluded_sector_ratio = 0.0f;
  float near_occlusion_ratio = 0.0f;
  std::vector<int> sector_point_histogram;
};

class OcclusionEvaluator
{
 public:
  explicit OcclusionEvaluator(const OcclusionEvaluatorConfig& config = OcclusionEvaluatorConfig())
      : config_(config),
        sector_width_(kTwoPi / static_cast<float>(config.sector_count))
  {
    if (config_.sector_count == 0)
    {
      config_.sector_count = 1;
      sector_width_ = kTwoPi;
    }
  }

  OcclusionEvaluation evaluate(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, double stamp_sec)
  {
    OcclusionEvaluation evaluation;
    if (!cloud || cloud->empty())
    {
      evaluation.state = OcclusionState::SEVERE_OCCLUSION;
      latest_state_ = evaluation.state;
      return evaluation;
    }

    std::vector<size_t> sector_counts(config_.sector_count, 0);
    std::vector<float> closest_distance(config_.sector_count, std::numeric_limits<float>::max());

    for (const auto& point : cloud->points)
    {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        continue;
      const float range_xy = std::sqrt(point.x * point.x + point.y * point.y);
      if (range_xy < 0.01f)
        continue;
      float angle = std::atan2(point.y, point.x);
      if (angle < 0.0f)
        angle += kTwoPi;
      const size_t idx = std::min(static_cast<size_t>(angle / sector_width_), config_.sector_count - 1);
      sector_counts[idx]++;
      if (range_xy < closest_distance[idx])
        closest_distance[idx] = range_xy;
    }

    evaluation.sector_point_histogram.reserve(sector_counts.size());
    for (const auto count : sector_counts)
      evaluation.sector_point_histogram.emplace_back(static_cast<int>(count));

    size_t occluded_sectors = 0;
    size_t near_occluded_sectors = 0;
    for (size_t i = 0; i < sector_counts.size(); ++i)
    {
      if (sector_counts[i] < config_.min_points_per_sector)
      {
        ++occluded_sectors;
        if (closest_distance[i] <= config_.near_distance_threshold)
          ++near_occluded_sectors;
      }
    }

    const float occluded_ratio = static_cast<float>(occluded_sectors) / static_cast<float>(config_.sector_count);
    const float near_ratio = occluded_sectors == 0
                                 ? 0.0f
                                 : static_cast<float>(near_occluded_sectors) / static_cast<float>(occluded_sectors);

    evaluation.occluded_sector_ratio = occluded_ratio;
    evaluation.near_occlusion_ratio = near_ratio;

    OcclusionState base_state = classifyStaticState(occluded_ratio, near_ratio);

    if (isDynamic(stamp_sec, base_state))
    {
      evaluation.state = OcclusionState::DYNAMIC_OCCLUSION;
    }
    else
    {
      evaluation.state = base_state;
    }

    latest_state_ = evaluation.state;
    latest_evaluation_ = evaluation;
    return evaluation;
  }

  OcclusionState latestState() const { return latest_state_; }
  const OcclusionEvaluation& latestEvaluation() const { return latest_evaluation_; }

 private:
  OcclusionState classifyStaticState(float occluded_ratio, float near_ratio) const
  {
    if (occluded_ratio <= 0.0f)
      return OcclusionState::NO_OCCLUSION;
    if (occluded_ratio <= config_.mild_ratio)
      return near_ratio > 0.5f ? OcclusionState::MODERATE_OCCLUSION : OcclusionState::MILD_OCCLUSION;
    if (occluded_ratio <= config_.moderate_ratio)
      return near_ratio > 0.6f ? OcclusionState::SEVERE_OCCLUSION : OcclusionState::MODERATE_OCCLUSION;
    if (occluded_ratio <= config_.severe_ratio)
      return near_ratio > 0.4f ? OcclusionState::SEVERE_OCCLUSION : OcclusionState::MODERATE_OCCLUSION;
    return OcclusionState::SEVERE_OCCLUSION;
  }

  bool isDynamic(double stamp_sec, OcclusionState candidate_state)
  {
    if (stamp_sec < 0.0)
      stamp_sec = 0.0;

    state_history_.emplace_back(stamp_sec, candidate_state);
    while (!state_history_.empty() &&
           (state_history_.back().first - state_history_.front().first) > config_.dynamic_time_window)
    {
      state_history_.pop_front();
    }

    size_t transitions = 0;
    for (size_t i = 1; i < state_history_.size(); ++i)
    {
      if (state_history_[i].second != state_history_[i - 1].second)
        ++transitions;
    }

    return transitions >= config_.dynamic_transition_threshold;
  }

  OcclusionEvaluatorConfig config_;
  float sector_width_;
  std::deque<std::pair<double, OcclusionState>> state_history_;
  OcclusionState latest_state_ = OcclusionState::NO_OCCLUSION;
  OcclusionEvaluation latest_evaluation_;
};

inline std::string toString(OcclusionState state)
{
  switch (state)
  {
    case OcclusionState::NO_OCCLUSION:
      return "No Occlusion";
    case OcclusionState::MILD_OCCLUSION:
      return "Mild Occlusion";
    case OcclusionState::MODERATE_OCCLUSION:
      return "Moderate Occlusion";
    case OcclusionState::SEVERE_OCCLUSION:
      return "Severe Occlusion";
    case OcclusionState::DYNAMIC_OCCLUSION:
      return "Dynamic Occlusion";
    default:
      return "Unknown";
  }
}

}  // namespace lidar_obstacle_detector
