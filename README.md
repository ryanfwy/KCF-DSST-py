# KCF-DSST-py
Python implementation of DSST tracking algorithm based on KCF tracker.

In [Baseline 3], the DSST scale estimation algorithm is added to the original KCF Tracker. Based on the python implementation of KCF Tracker, see [Baseline 2], the code of DSST is translated from C++ and added to the KCF in python.

## Requirements
- Python 2.7 (or 3)
- NumPy
- Numba (needed if you want to use the hog feature)
- OpenCV (ensure that you can import cv2 in python)

## Baseline
Some implementations of KCF and DSST algorithms.

### 1. KCF Tracker in C++
[C++ KCF Tracker](https://github.com/joaofaro/KCFcpp): Original C++ implementation of Kernelized Correlation Filter (KCF) [1, 2].

### 2. KCF Tracker in Python
[KCF tracker in Python](https://github.com/uoip/KCFpy): Python implementation of KCF Tracker.

### 3. DSST Tracker in C++
[KCF-DSST](https://github.com/liliumao/KCF-DSST): C++ implementation of Discriminative Scale Space Tracker (DSST) [3].

## Reference
[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

[3] M. Danelljan, G. Häger, F. Shahbaz Khan, and M. Felsberg. "Accurate scale estimation for robust visual tracking". In Proceedings of the British Machine Vision Conference (BMVC), 2014.
