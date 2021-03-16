#! /bin/bash

# Download the weights
wget https://cloud.orenbell.com/index.php/s/6eGA6mRyKSeHFNp/download -O models_weights.zip 
unzip models_weights.zip

# Extract obj det AI model
cp yolo4_deer/yolo4_deer.cpp ros2/src/cinematography/tkDNN/tests/darknet/
cp yolo4_deer/yolo4_deer.cfg ros2/src/cinematography/tkDNN/tests/darknet/cfg/
cp yolo4_deer/yolo4_deer.names ros2/src/cinematography/tkDNN/tests/darknet/names/
mkdir -p ros2/src/cinematography/tkDNN/build/yolo4_deer
cp -r yolo4_deer/debug ros2/src/cinematography/tkDNN/build/yolo4_deer/
cp -r yolo4_deer/layers ros2/src/cinematography/tkDNN/build/yolo4_deer/

# Build obj det trt engine
cd ros2/src/cinematography/tkDNN/build
cmake ..
make -j $(nproc)
./test_yolo4_deer
cp yolo4_deer_fp32.rt ../../../../../
cd ../../../../../

# Build hde trt engine
cd trt_builder
make
./build_trt hde.onnx input_0.pb deer_hde_fp32.rt
cp deer_hde_fp32.rt ../

# Cleanup
cd ../
rm -r yolo4_deer trt_builder models_weights.zip