CUR_DIR=`pwd`
CMAKE_DIR="/workspace/xys/cmake-3.24.3-linux-x86_64/bin"
# rm -rf build
mkdir -p build
cd ${CUR_DIR}/build
$CMAKE_DIR/cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE=OFF \
	-DCMAKE_INSTALL_PREFIX=${CUR_DIR}/install \
	-DBUILD_TF=OFF \
	-DBUILD_PYT=OFF \
	-DBUILD_TRT=OFF \
	-DBUILD_MULTI_GPU=OFF \
	-DUSE_NVTX=OFF \
	-DBUILD_EXAMPLE=OFF \
	-DBUILD_TEST=OFF \
	-DBUILD_TRT=ON \
	-DBUILD_ORGIN_NET=OFF \
	..


make -j$(nproc) 

cp ${CUR_DIR}/build/lib/libErniePlugin.so ${CUR_DIR}/../../so/plugins/libErniePlugin.so

