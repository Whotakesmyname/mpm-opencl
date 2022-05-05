# Material Point Method with OpenCL

This project implements an MPM solver with OpenCL. It is still under development.

## Build Instructions
This project includes 2D and 3D versions of MPM implementation. You can find the corresponding CMake target
**mpm-opencl-2d** and **mpm-opencl-3d** respectively.

To build this project, you can simply
```bash
mkdir build
cd build
cmake ..
make mpm-opencl-2d
make mpm-opencl-3d
```

## License and Copyright
Licensed under MIT License. This project uses OpenCL-SDK as a codebase, especially the OpenGL & OpenCL interoperation feature demonstrated in N-Body sample.