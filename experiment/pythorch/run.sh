# /bin/sh
cmake -B build
cmake --build build --config Release --parallel
cd build/src 
./example-app
