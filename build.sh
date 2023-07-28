mkdir -p build
WORKDIR=`pwd`
cd build
echo $WORKDIR/Halide-14.0.0-x86-64-linux/lib/cmake/HalideHelpers
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$WORKDIR/Halide-14.0.0-x86-64-linux/lib/cmake/" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$WORKDIR/Halide-14.0.0-x86-64-linux/lib/cmake/HalideHelpers:$WORKDIR/Halide-14.0.0-x86-64-linux/lib/cmake/Halide" ..
make
