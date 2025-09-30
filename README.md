# 🎯 Gaussian Mixture Model (GMM) via Expectation–Maximization (C++17)

This project implements a **Gaussian Mixture Model** from scratch in modern **C++17**, using the **Expectation–Maximization (EM)** algorithm.  
It’s designed as a **teaching / learning project** — minimal dependencies, clean structure, and includes a Python visualization script.

---

## 🚀 Features

- ✅ Full GMM with **E-step / M-step** loop  
- ✅ **Log-likelihood monitoring** each iteration  
- ✅ **Cluster interpretation**: means, covariances, weights  
- ✅ Synthetic 2D dataset generator (great for demos)  
- ✅ CSV I/O for real datasets  
- ✅ Visualization using Python + Matplotlib  
- ✅ No external C++ libraries required

---

## 🛠️ Build & Run

### **1. Compile**
```bash
g++ -O2 -std=c++17 gmm_em.cpp -o gmm

### **2. RUN on synthetic data**
```bash
./gmm --k 3 --max_iters 200 --tol 1e-6

### **3. RUN on your own dataset**
```bash
./gmm --input data.csv --k 4 --save results.csv


