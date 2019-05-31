# Robust Estimation of Similarity Transformation for Visual Object Tracking

This is a C++ implementation of the Large Displacement Estimation of Similarity transformation (LDES) tracker. The main idea is to extend the CF-based tracker with similarity transformation (including position, scale, and rotation) in an efficient way. The details can be found in the [AAAI-2019 paper](https://arxiv.org/abs/1712.05231).

# Instruction
* Modify the path in the function testLDES in main.cpp, which takes the image file path and the label path as input. The .txt files can be generated using our gen_filelist.py script.
* mkdir build && cd build && cmake .. && make all -j32 
* Run the code: ./LDESTracker

# Dependency
OpenCV>=3.3.0. We test the code on OpenCV-3.1.0, which shows different results. This is weird, so OpenCV>=3.3.0 is recommended, for less trouble..

# Notes
There are some differences between the C++ code and the [MATLAB code](https://github.com/ihpdep/LDES). They are listed below.
* The block gradient descend (BGD) is not implemented in our code, for higher efficiency. Our code runs correlation filter and phase correlation only one time.
* The feature map fusion is not implemented in our code. We use the fhog feature map. Thanks to [KCFCpp](https://github.com/joaofaro/KCFcpp). We take part from your code!
* The ugly code in main.cpp is kept for a better vision on the connection between LDES and KCF, and tell you how to use phase correlation to estimate the scale and rotation.

# Example
![tracking-example][logo]

[logo]: https://github.com/yfji/LDESCpp/blob/square/examples/toy.gif "tracking-example"

# Contact 
* Yufeng Ji, jyf131462@163.com
