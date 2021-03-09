  
struct Conv2DParams{

  int x_height;
  int x_width;
  int x_channels;

  int k_height;
  int k_width;
  int k_channels;

  int k_dilation_h;
  int k_dilation_v;

  int k_stride_h;
  int k_stride_v;

};

class Conv2DIterator {

  const Conv2DParams hyper_param_min;
  const Conv2DParams hyper_param_max;

  public:

    Conv2DIterator(Conv2DParams &param_min, Conv2DParams &param_max);
};

Conv2DIterator::Conv2DIterator(Conv2DParams &param_min, Conv2DParams &param_max):
  hyper_param_min(param_min), hyper_param_max(param_max) {

}



