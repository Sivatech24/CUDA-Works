#include <iostream>
#include <cuda_runtime.h>

#define N 64  // 64x64 matrix

// CUDA Kernel for operations
__global__ void addMatrix(float *A, float *B, float *C) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int idx = row * N + col;
    C[idx] = A[idx] + B[idx];
}

__global__ void subMatrix(float *A, float *B, float *C) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int idx = row * N + col;
    C[idx] = A[idx] - B[idx];
}

__global__ void mulMatrix(float *A, float *B, float *C) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int idx = row * N + col;
    C[idx] = A[idx] * B[idx];
}

__global__ void divMatrix(float *A, float *B, float *C) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int idx = row * N + col;
    if (B[idx] != 0) // Prevent division by zero
        C[idx] = A[idx] / B[idx];
    else
        C[idx] = 0;
}

int main() {
    int size = N * N * sizeof(float);

    // Allocate host memory
    float h_A[N][N], h_B[N][N], h_C[N][N];

    // Initialize matrices with some values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i][j] = i + j;
            h_B[i][j] = (i - j) ? (i - j) : 1;  // Avoid division by zero
        }
    }

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define CUDA execution configuration
    dim3 threadsPerBlock(N, N);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Addition
    cudaEventRecord(start);
    addMatrix<<<1, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float timeAdd;
    cudaEventElapsedTime(&timeAdd, start, stop);
    std::cout << "Addition completed in: " << timeAdd << " ms\n";

    // Subtraction
    cudaEventRecord(start);
    subMatrix<<<1, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float timeSub;
    cudaEventElapsedTime(&timeSub, start, stop);
    std::cout << "Subtraction completed in: " << timeSub << " ms\n";

    // Multiplication
    cudaEventRecord(start);
    mulMatrix<<<1, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float timeMul;
    cudaEventElapsedTime(&timeMul, start, stop);
    std::cout << "Multiplication completed in: " << timeMul << " ms\n";

    // Division
    cudaEventRecord(start);
    divMatrix<<<1, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float timeDiv;
    cudaEventElapsedTime(&timeDiv, start, stop);
    std::cout << "Division completed in: " << timeDiv << " ms\n";

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
