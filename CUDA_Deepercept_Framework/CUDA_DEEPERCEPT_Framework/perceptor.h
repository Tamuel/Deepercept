#ifndef PERCEPTOR_H
#define PERCEPTOR_H

#include "base.h"
#include "tensor.h"
#include "srcMeasure.h"
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

	// Convolution forward algorithm preferance
	cudnnConvolutionFwdAlgo_t convFwdAlg;

	// Dummy tensor for matrix calculation efficiency
	Tensor* dummyTensor;

	// Set true if you want to debug and carry about limit condition of operations, if you don't need that then set false
	bool debugMode;

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

	void setDevice(int aDeviceId) {
		cudaSetDevice(aDeviceId);
	}

	void setDevice() {
		cudaSetDevice(mDeviceId);
	}

public:
	Perceptor(int aDeviceId = 0, bool aDebugMode = true) {
		if (gpuUtilization[aDeviceId] == true) {
			cout << "GPU" << aDeviceId << " already assigned" << endl;
			exit(EXIT_FAILURE);
		}

		debugMode = aDebugMode;

		mDeviceId = aDeviceId;
		gpuUtilization[mDeviceId] = true;

		cudaSetDevice(mDeviceId);
		CuBLAS_ERROR(cublasCreate(&cuBlasHandle));
		CuDNN_ERROR(cudnnCreate(&cuDnnHandle));

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

	// Set random value from min to max with variance var
	void matRand(Tensor* tA, dtype min, dtype max);

	// Execute convolution operation with tInput tensor and tFilter tensor. And then return pointer of this perceptor
	Perceptor* convolution(Tensor* tInput, Tensor* tFilter, Tensor* tOutput);

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

	void testNetwork() {
		Tensor bias({ 1, 1, 3, 3 }, "temp", 2.3); // NCHW
		Tensor dest({ 1, 1, 3, 3 }, "temp2", 2.5); // NCHW
		Tensor dest2({ 1, 1, 3, 3 }, "Conv1_input", 0); // NCHW
		cudnnTensorDescriptor_t t_desc;
		SrcMeasure sm;
		sm.startTime(0);
		cudnnCreateTensorDescriptor(&t_desc);
		//printf("%d %d %d %d\n", bias.shape(0), bias.shape(1), bias.shape(2), bias.shape(3));
		cudnnSetTensor4dDescriptor(t_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bias.shape(0), bias.shape(1), bias.shape(2), bias.shape(3));
		//cudnnSetTensor4dDescriptorEx()
		sm.endTime(0, "Set descriptor");
		//cudnnTransformTensor()
		sm.startTime(0);
		sendToDevice(&bias);
		sm.endTime(0, "Send bias");
		sm.startTime(0);
		sendToDevice(&dest);
		sm.endTime(0, "Send dest");
		float value = 3.5;
		cudnnSetTensor(
			cuDnnHandle,
			t_desc,
			dest.devDataPtr(),
			&value
		);

		float alpha = 1;
		CuDNN_ERROR(
			cudnnAddTensor( // Add bias to dest
				cuDnnHandle,
				&alpha, // Coefficient
				t_desc, // Bias tensor descriptor
				bias.devDataPtr(), // Bias tensor data pointer
				&alpha, // Coefficient
				t_desc, // Destination tensor descriptor
				dest.devDataPtr() // Destination tensor data
			)
		);

		sendToDevice(&dest2);
		cudnnOpTensorDescriptor_t t_op_desc;
		cudnnCreateOpTensorDescriptor(&t_op_desc);
		cudnnSetOpTensorDescriptor(t_op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);

		float beta = 0;
		CuDNN_ERROR(
			cudnnOpTensor(
				cuDnnHandle,
				t_op_desc,
				&alpha,
				t_desc,
				bias.devDataPtr(),
				&alpha,
				t_desc,
				dest.devDataPtr(),
				&beta,
				t_desc,
				dest2.devDataPtr()
			)
		);

		
		float scale = 1.5;
		CuDNN_ERROR(
			cudnnScaleTensor(
				cuDnnHandle,
				t_desc,
				dest2.devDataPtr(),
				&scale
			)
		);

		Tensor conv_filter1({ 1, 1, 3, 3 }, "Conv_filter", 0);

		cudnnFilterDescriptor_t filter_desc;
		cudnnCreateFilterDescriptor(&filter_desc);
		cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv_filter1.shape(0), conv_filter1.shape(1), conv_filter1.shape(2), conv_filter1.shape(3));
	
		cudnnConvolutionDescriptor_t conv_desc;
		cudnnCreateConvolutionDescriptor(&conv_desc);
		CuDNN_ERROR(
			cudnnSetConvolution2dDescriptor(
				conv_desc, // convDesc
				1, // pad_h
				1, // pad_w
				1, // u
				1, // v
				1, // dialation_h
				1, // dialation_w
				CUDNN_CROSS_CORRELATION // convolutionMode
			)
		);

		int n, c, h, w;
		cudnnGetConvolution2dForwardOutputDim(
			conv_desc,
			t_desc,
			filter_desc,
			&n,
			&c,
			&h,
			&w
		);

		Tensor conv1({ n, c, h, w }, "Conv1", 0);
		conv1.printShape();
		sm.startTime(1);
		cudnnTensorDescriptor_t conv1_desc;
		cudnnCreateTensorDescriptor(&conv1_desc);
		cudnnSetTensor4dDescriptor(conv1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
		sm.endTime(1, "Descriptor create time");


		sm.startTime(0);
		sendToDevice(&conv_filter1);
		matRand(&conv_filter1, -1, 1);
		sm.endTime(0, "Send filter");
		sm.startTime(0);
		sendToDevice(&conv1);
		sm.endTime(0, "Send conv");
		sm.startTime(1);
		int returnedAlgoCount[3];
		cudnnConvolutionFwdAlgoPerf_t conv_perf[3];
		cudnnFindConvolutionForwardAlgorithm(
			cuDnnHandle,
			t_desc,
			filter_desc,
			conv_desc,
			conv1_desc,
			3,
			returnedAlgoCount,
			conv_perf
		);
		sm.endTime(1, "Get prefer convolution algorithm");

		Tensor conv_input({ 1, 1, 3, 3 }, "Conv_input", 0);
		sendToDevice(&conv_input);
		dtype k = 1;
		cudnnSetTensor(
			cuDnnHandle,
			t_desc,
			conv_input.devDataPtr(),
			&k
		);

		retrievDataFromDevice(&conv_filter1);
		conv_filter1.print(true);
		conv_filter1.print2();

		retrievDataFromDevice(&conv_input);
		conv_input.print(true);
		conv_input.print2();
		alpha = 1; beta = 0;
		CuDNN_ERROR(
			cudnnConvolutionForward( // Column major approach
				cuDnnHandle, // Handle
				&alpha, // Alpha : Input tensor coefficient
				t_desc, // xDesc
				conv_input.devDataPtr(), // *x
				filter_desc, // wDesc
				conv_filter1.devDataPtr(), // *w
				conv_desc, // convDesc
				CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, // Algo
				NULL, // *workSpace
				0, // workSpaceSizeInBytes
				&beta, // *beta : Output tensor coefficient
				conv1_desc, // yDesc
				conv1.devDataPtr() // *y
			)
		);

		retrievDataFromDevice(&conv1);
		conv1.print(true);
		conv1.print2();

		cudnnDestroyTensorDescriptor(t_desc);
		cudnnDestroyOpTensorDescriptor(t_op_desc);
		cudnnDestroyFilterDescriptor(filter_desc);
		cudnnDestroyConvolutionDescriptor(conv_desc);
	}
};

#endif