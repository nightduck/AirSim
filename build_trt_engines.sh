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

# Extract yolo4 tiny obj detection model
cp yolo4tiny_deer/yolo4tiny_deer.cpp ros2/src/cinematography/tkDNN/tests/darknet/
cp yolo4tiny_deer/yolo4tiny_deer.cfg ros2/src/cinematography/tkDNN/tests/darknet/cfg/
cp yolo4tiny_deer/yolo4tiny_deer.names ros2/src/cinematography/tkDNN/tests/darknet/names/
mkdir -p ros2/src/cinematography/tkDNN/build/yolo4tiny_deer
cp -r yolo4tiny_deer/debug ros2/src/cinematography/tkDNN/build/yolo4tiny_deer/
cp -r yolo4tiny_deer/layers ros2/src/cinematography/tkDNN/build/yolo4tiny_deer/

# Build yolo4 tiny trt engines
cd ros2/src/cinematography/tkDNN/build
cmake ..
make -j $(nproc)
./test_yolo4_deer
./test_yolo4tiny_deer yolo4tiny_deer ../tests/darknet/cfg/yolo4tiny_deer.cfg ../tests/darknet/names/yolo4tiny_deer.names
cp yolo4_deer_fp32.rt ../../../../../
cp yolo4tiny_deer_fp32.rt ../../../../../
cd ../../../../../

# Build hde trt engine
cd trt_builder
make
./build_trt hde.onnx input_0.pb deer_hde_fp32.rt
cp deer_hde_fp32.rt ../

# Cleanup
cd ../
rm -r yolo4_deer yolo4tiny_deer trt_builder models_weights.zip