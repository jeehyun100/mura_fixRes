# FixRes for mura


# Installation

Install
* pip install -r requirements.txt

# Download Mura DataSet
* ./MURA-V1.1

# Using the code 

* restnet152
    * python main_resnet152_scratch.py
* efficientnet b7
    * python main_effi_b7_scratch.py

# Test
* Uncomment test method in main_xxx_scratch.py and comment run method.

# What i Did
* Resnet 156
* Efficient B7
* fixres Train size 320,  test size 320 * 1.14
* Histogram Equalization https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html
