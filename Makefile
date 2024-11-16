BUILD_EXAMPLES=OFF
BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)
USE_GPU=OFF

default:
	@cmake -B build
	@cmake --build build --config Release --parallel
	@cd build/src
	@./inference 

clean:
	@rm -rf build*

