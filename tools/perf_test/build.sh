rm -rf build
mkdir build
cd build
cmake -DRUN_TYPE=FULLHAN -DCMAKE_BUILD_TYPE=Debug ..
make
