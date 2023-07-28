#include <Halide.h>

using namespace Halide;

class ValueSpreadGenerator : public Halide::Generator<ValueSpreadGenerator>
{
public:
    Input<Buffer<uint8_t>> input{"input", 2};
    Output<Buffer<uint8_t>> output{"spread", 2};

    void generate()
    {
        Var x("x"), y("y");
        Func clamped;
        Expr x_clamped = clamp(x, 0, input.width() - 1);
        Expr y_clamped = clamp(y, 0, input.height() - 1);
        clamped(x, y) = input(x_clamped, y_clamped);

        RDom box(-2, 5, -2, 5);
        output(x, y) = (maximum(clamped(x + box.x, y + box.y)) -
                        minimum(clamped(x + box.x, y + box.y)));

        Var yo, yi;
        output.split(y, yo, yi, 32).parallel(yo);
        output.vectorize(x, 16);
        clamped.store_at(output, yo).compute_at(output, yi);
    }
};

HALIDE_REGISTER_GENERATOR(ValueSpreadGenerator, value_spread_generator);

