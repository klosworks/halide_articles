# halide_articles

This is the source code for the article https://piotrarturklos.vision/2022/08/05/halide-why-you-must-know-it-and-why-others-will-keep-ignoring-it/

Also, it can be treated as a Hello World example, for starting projects that want to use Halide the right way, that is with ahead-of-time compilation, generators and CMake. This is only for Linux target though.

# build instructions

The setup.sh command downloads an official Halide release and unpacks it. The build command does complete building, including binaries that generate libraries using Halide, and binaries that use those libraries.

```
./setup.sh
./build.sh
```

# running 

Just run one of the binaries in the _build_ folder.
