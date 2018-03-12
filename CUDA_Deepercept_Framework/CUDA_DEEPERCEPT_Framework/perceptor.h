#ifndef PERCEPTOR_H
#define PERCEPTOR_H

#include "basics.h"
#include "tensor.h"
#include <cublas.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cudnn.h>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define MATRIX_DIM_LIMIT 4096
#define N_MAXIMUM_GPU 4


class Perceptor{
private:
	// Store which GPU is now utilize
	static bool gpuUtilization[N_MAXIMUM_GPU];

	// Handle for cuBals
	cublasHandle_t cuBlasHandle;
	// Handle for cuDNN
	cudnnHandle_t cuDnnHandle;

	// Dummy tensor for matrix calculation efficiency
	Tensor* dummyTensor;

	// Specific GPU ID for this perceptor
	int mDeviceId;

	// To synchronize GPU stream or not
	bool synchronizeStream;

	void CuBLAS_ERROR(cublasStatus_t error) {
		switch (error) {
		case CUBLAS_STATUS_SUCCESS:
			return;
		case CUBLAS_STATUS_NOT_INITIALIZED:
			cerr << "cuBLAS : Library was not initialized." << endl;
			break;
		case CUBLAS_STATUS_ALLOC_FAILED:
			cerr << "cuBLAS : Resource allocation failed." << endl;
			break;
		case CUBLAS_STATUS_INVALID_VALUE:
			cerr << "cuBLAS : An unsupported value or parameter was passed to the function." << endl;
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			cerr << "cuBLAS : The function requires a feature absent from the device architecture;" <<
				" usually casued by the lack of support for double precision." << endl;
			break;
		case CUBLAS_STATUS_MAPPING_ERROR:
			cerr << "cuBLAS : An access to GPU memory space failed." << endl;
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			cerr << "cuBLAS : The GPU program failed to execute." << endl;
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR:
			cerr << "cuBLAS : An internal cuBLAS operation failed." << endl;
			break;
		case CUBLAS_STATUS_NOT_SUPPORTED:
			cerr << "cuBLAS : The functionnality requested is not supported." << endl;
			break;
		case CUBLAS_STATUS_LICENSE_ERROR:
			cerr << "cuBLAS : The functionnality requested requires some license and an error" <<
				" was detected when trying to check the current licensing." << endl;
			break;
		}
		cerr << " File : " << __FILE__ << ", Line : " << __LINE__ << endl;
		exit(EXIT_FAILURE);
	}

	void CuDNN_ERROR(cudnnStatus_t error) {
		switch (error) {
		case CUDNN_STATUS_SUCCESS:
			return;
		case CUDNN_STATUS_NOT_INITIALIZED:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : Library was not initialized." << endl;
			break;
		case CUDNN_STATUS_INVALID_VALUE:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An incorrect value or parameter was passed." << endl;
			break;
		case CUDNN_STATUS_ALLOC_FAILED:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : Resource allocation failed." << endl;
			break;
		case CUDNN_STATUS_BAD_PARAM:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An incorrect value or parameter was passed." << endl;
			break;
		case CUDNN_STATUS_ARCH_MISMATCH:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The function requires a feature absent from the device architecture;" <<
				" usually casued by the lack of support for double precision." << endl;
			break;
		case CUDNN_STATUS_MAPPING_ERROR:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An access to GPU memory space failed." << endl;
			break;
		case CUDNN_STATUS_EXECUTION_FAILED:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The GPU program failed to execute." << endl;
			break;
		case CUDNN_STATUS_INTERNAL_ERROR:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An internal cuBLAS operation failed." << endl;
			break;
		case CUDNN_STATUS_NOT_SUPPORTED:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The functionnality requested is not supported." << endl;
			break;
		case CUDNN_STATUS_LICENSE_ERROR:
			cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The functionnality requested requires some license and an error" <<
				" was detected when trying to check the current licensing." << endl;
			break;
		}
		cerr << " File : " << __FILE__ << ", Line : " << __LINE__ << endl;
		exit(EXIT_FAILURE);
	}

	void syncGpuStream() {
		if (synchronizeStream) {
			cudaDeviceSynchronize();
		}
	}

	void setDevice() {
		cudaSetDevice(mDeviceId);
	}

public:
	Perceptor(int aDeviceId = 0) {
		if (gpuUtilization[aDeviceId] == true) {
			cout << "GPU" << aDeviceId << " already assigned" << endl;
			exit(EXIT_FAILURE);
		}

		mDeviceId = aDeviceId;
		gpuUtilization[mDeviceId] = true;

		cudaSetDevice(mDeviceId);
		cublasCreate(&cuBlasHandle);
		cudnnCreate(&cuDnnHandle);

		synchronizeStream = true;
		dummyTensor = new Tensor({ MATRIX_DIM_LIMIT, MATRIX_DIM_LIMIT });
		dummyTensor->setDevice(mDeviceId);

		sendToDevice(dummyTensor);
	}

	~Perceptor() {
		gpuUtilization[mDeviceId] = false;
		cublasDestroy(cuBlasHandle);
		cudnnDestroy(cuDnnHandle);
	}

	void setSynchronizeGpuStream(bool aSync) {
		synchronizeStream = aSync;
	}

	// Check tensor tA and tB are in same device
	void checkDevice(Tensor* tA, Tensor* tB);
	// Check device ID of perceptor and device ID of tensor tB are same
	void checkDevice(Tensor* tB);

	// Matrix operations
	// Return = alpha * ( tA x tB ) + beta * Out
	Tensor* matSgemm(Tensor* tA, Tensor* tB, float alpha, float beta);
	// tOut = alpha * ( tA x tB ) + beta * Out
	void matSgemm(Tensor* tOut, Tensor* tA, Tensor* tB, float alpha, float beta);

	// Return = tA x tB
	Tensor* matMult(Tensor* tA, Tensor* tB);
	// tOut = tA x tB
	void matMult(Tensor* tOut, Tensor* tA, Tensor* tB);

	// tB = (Scalar) scalA * (Tensor) tB
	void matMult(dtype scalA, Tensor* tB);

	// Return = alpha * tA + beta * tB
	Tensor* matSgeam(Tensor* tA, Tensor* tB, float alpha, float beta);
	// tOut = alpha * tA + beta * tB
	void matSgeam(Tensor* tOut, Tensor* tA, Tensor* tB, float alpha, float beta);

	// Return = tA + tB
	Tensor* matAdd(Tensor* tA, Tensor* tB);
	// tOut = tA + tB
	void matAdd(Tensor* tOut, Tensor* tA, Tensor* tB);

	// Return = tA - tB
	Tensor* matSub(Tensor* tA, Tensor* tB);
	// tOut = tA - tB
	void matSub(Tensor* tOut, Tensor* tA, Tensor* tB);

	// Return = tA * tB (Haramard Product)
	Tensor* matEltMult(Tensor* tA, Tensor* tB);
	// tOut = tA * tB (Haramard Product)
	void matEltMult(Tensor* tOut, Tensor* tA, Tensor* tB);

	// Swap two tensors, tA = tB, tB = tA.
	// If you want to swap tA and tB with different shape but same size change forceSwap to true.
	void matSwap(Tensor* tA, Tensor* tB, bool forceSwap = false);

	// Copy tensor tA to tB
	void matCopy(Tensor* tB, Tensor* tA);

	// Get index (int) of maximum element
	int matMaxIndex(Tensor* tA);

	// Get index (int) of minimum element
	int matMinIndex(Tensor* tA);

	// Transpose matrix
	void matTranspose(Tensor* tA);
	
	// (Scalar) Return = Sum of matrix elements
	dtype matSum(Tensor* tA);

	// The other operations
	void getGpuInformation();

	void getCuDnnVersion();

	void getCuBlasVersion();

	void getGpuDriverVersion();

	int deviceId() {
		return mDeviceId;
	}

	// Allocate tensor t to device and return device pointer of allocated tensor
	void sendToDevice(Tensor* t, bool sendData = true);

	// Allocate tensor t data to device and return device pointer of allocated data
	void sendDataToDevice(Tensor* t);

	// Retrieve tensor t data from tensor device pointer
	void retrievDataFromDevice(Tensor* t, bool retreiveOnlyData = true);
};

#endif