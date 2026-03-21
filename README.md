# 3-Coloring of Grid Graphs: Recurrence Order Calculator
C++ transfer-matrix algorithm with Berlekamp-Massey for the 3-coloring of grid graphs.

This repository contains a highly optimized C++ implementation for calculating the degree of the minimal polynomial (order of linear recurrence) for the transfer matrix of the 3-coloring problem on $W \times L$ grid graphs.

This codebase provides the empirical computational data supporting the connection between the $q=3$ antiferromagnetic Potts model transfer matrix and the restricted Motzkin paths defined in **OEIS A005717**.

## How to Run

This algorithm is heavily optimized. It utilizes **Compressed Sparse Row (CSR)** formats for maximum L1/L2 cache locality, eliminates atomic operations during Krylov iterations, and leverages **OpenMP** for parallel processing. 

To achieve the intended maximum performance, you **must** compile the code with the appropriate optimization flags.

### 1. Compile
Please run the following command in your terminal. Ensure you have GCC installed with OpenMP support.

```bash
g++ -O3 -march=native -fopenmp main.cpp -o main
```
#### Flag Explanations:
##### -O3: Enables the highest level of compiler optimization.
##### -march=native: Tells the compiler to use your CPU's specific architecture (crucial for unlocking the AVX2 and BMI2 #pragma directives inside the code).
##### -fopenmp: Enables OpenMP multi-threading for the Phase 2 matrix iterations.

### 2. Execute
```bash
.\main
```
## Expected Output
```bash
1 1 2 3 5 7 13 19 35 51 96 141 267 393 750 1107 2123 3139 6046 8953
```

### Algorithmic Highlights
#### Dynamic Bit-width Packing: States are natively packed into uint64_t using 2-bit color representations to maximize memory bandwidth efficiency.
#### Isomorphic Reduction: Uses a get_canonical mapping function to collapse the $3 \cdot 2^{W-1}$ raw state space into a minimal independent quotient space, exploiting the $S_3$ global color permutation symmetry.
#### CSR Graph Flattening: Transforms the pull-based transition graph into a Compressed Sparse Row (CSR) format. This ensures maximum L1/L2 cache locality and allows for completely lock-free, atomic-free OpenMP multithreading during the profile DP iterations.
#### $O(N^2)$ Recurrence Extraction: Bypasses the $O(V^3)$ bottleneck of standard matrix exponentiation. By utilizing randomized initial state weights, it generates the Krylov sequence and extracts the exact degree of the minimal polynomial using the Berlekamp-Massey algorithm.
