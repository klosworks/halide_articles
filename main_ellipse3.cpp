#include <Halide.h>
#include <opencv2/opencv.hpp>
int main()
{
    using namespace Halide;
    Func gradient;
    Var x, y, c;
    Expr cx = cast<float>(x) * x;
    Expr cy = cast<float>(y) * y;
    float a_sq = 200.f * 200.f, b_sq = 150.f * 150.f;
    gradient(x, y, c) = cast<uint8_t>(cast<int32_t>(
                (cx / a_sq + cy / b_sq) * 255.f + c * 85.f) & 0xff);
    auto output = Halide::Runtime::Buffer<uint8_t>::make_interleaved(320, 240, 3);
    gradient.output_buffer().dim(0).set_stride(3);
    gradient.realize(output);
    cv::Mat_<cv::Vec3b> wrapper(240, 320, (cv::Vec3b*)output.data());
    cv::imwrite("gradient.jpg", wrapper);
}
