cd ../deps/googletest
mkdir -p build
cd build
cmake CXXFLAGS="-pthread" ../ 
make all
