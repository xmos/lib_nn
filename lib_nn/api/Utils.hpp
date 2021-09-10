#pragma once
#include <cstdint>

#define fetch_int(buf)                                 \
  (((uint8_t)buf[3] << 24) + ((uint8_t)buf[2] << 16) + \
   ((uint8_t)buf[1] << 8) + ((uint8_t)buf[0] << 0))

namespace nn {

inline bool is_aligned(const void* ptr, std::uintptr_t alignment) noexcept {
  auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  return !(iptr % alignment);
}

/**
 * This class ultimately just decorates subclasses with some static fields and
 * methods related to its channel-wise parallelism
 */
template <int N_channels_per_cog_log2>
class ChannelParallelComponent {
 public:
  /**
   * log2() of the number of channels processed in parallel.
   */
  static constexpr int ChannelsPerOutputGroupLog2 = N_channels_per_cog_log2;
  /**
   * The number of channels processed in parallel.
   */
  static constexpr int ChannelsPerOutputGroup =
      (1 << ChannelsPerOutputGroupLog2);

  /**
   * Get the number of output channel groups associated with an output image.
   */
  static int OutputGroups(const int output_channels) {
    return (output_channels + ChannelsPerOutputGroup - 1) >>
           ChannelsPerOutputGroupLog2;
  }
};

template <int N_channels_per_cog_log2>
constexpr int ChannelParallelComponent<
    N_channels_per_cog_log2>::ChannelsPerOutputGroupLog2;
template <int N_channels_per_cog_log2>
constexpr int
    ChannelParallelComponent<N_channels_per_cog_log2>::ChannelsPerOutputGroup;

}  // namespace nn
