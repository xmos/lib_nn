cd ../deps/googletest
mkdir -p build
cd build
cmake CXXFLAGS="-pthreads" ../ 
make all
