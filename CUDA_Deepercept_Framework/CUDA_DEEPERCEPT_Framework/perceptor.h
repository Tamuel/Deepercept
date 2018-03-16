#ifndef PERCEPTOR_H
#define PERCEPTOR_H

#include "base.h"
#include "tensor.h"
#include "layer.h"
#include "srcMeasure.h"
#include "layer.h"
#include <memory>
#include <vector>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define MATRIX_DIM_LIMIT 4096
#define N_MAXIMUM_GPU 4

struct threadBlocks {
	dim3 threads;
	dim3 blocks;
};
typedef struct threadBlocks tBlocks;

class Perceptor{
private:
	// Store which GPU is now utilize
	static bool GPU_UTILIZATION[N_MAXIMUM_GPU];

	// Handle for cuBals
	cublasHandle_t mCublasHandle;

	// Handle for cuDNN
	cudnnHandle_t mCudnnHandle;

	// Layer vector for deeplearning
	//vector<unique_ptr<Layer>> mLayers;

	// Dummy tensor for matrix calculation efficiency
	Tensor* dummyTensor;

	// Set true if you want to debug and carry about limit condition of operations, if you don't need that then set false
	bool mDebugMode;

	// Specific GPU ID for this perceptor
	int mDeviceId;

	// To synchronize GPU stream or not
	bool mSynchronizeStream;

	void syncGpuStream() {
		if (mSynchronizeStream) {
			cudaDeviceSynchronize();
		}
	}

	// Change device which have [aDeviceId] ID
	void changeDevice(int aDeviceId) {
		cudaSetDevice(aDeviceId);
	}

	// Change device which allocated to current perceptor
	void changeDevice() {
		cudaSetDevice(mDeviceId);
	}

	// Get number of threads and thread blocks for handle tensor tA
	tBlocks getThreadBlocks(Tensor* tA);

	void debugOut(string s) {
		if (mDebugMode) cout << "Perceptor" << deviceId() << "[" << s << "]" << endl;
	}

public:
	Perceptor(int aDeviceId = 0, bool aDebugMode = true) {
		if (GPU_UTILIZATION[aDeviceId] == true) {
			cout << "GPU" << aDeviceId << " already assigned" << endl;
			exit(EXIT_FAILURE);
		}

		mDebugMode = aDebugMode;

		mDeviceId = aDeviceId;
		GPU_UTILIZATION[mDeviceId] = true;
		mSynchronizeStream = true;

		cudaSetDevice(mDeviceId);
		debugOut("Set device");

		CuBLAS_ERROR(cublasCreate(&mCublasHandle));
		debugOut("Initialize cuBlas");

		CuDNN_ERROR(cudnnCreate(&mCudnnHandle));
		debugOut("Initialize cuDNN");

		debugOut("Create dummy tensor for calculation");
		dummyTensor = new Tensor({ MATRIX_DIM_LIMIT, MATRIX_DIM_LIMIT });
		dummyTensor->setDevice(mDeviceId);
		sendTensorToDevice(dummyTensor);

		debugOut("Perceptor" + to_string(aDeviceId) + " initialized");
	}

	~Perceptor() {
		GPU_UTILIZATION[mDeviceId] = false;
		cublasDestroy(mCublasHandle);
		cudnnDestroy(mCudnnHandle);
	}

	void setSynchronizeGpuStream(bool aSync) {
		mSynchronizeStream = aSync;
	}

	// Check device ID of perceptor and device ID of tensor tA are same
	void checkDeviceAndAllocate(Tensor* tA);

	// Matrix operations
	// Return = alpha * ( tA x tB ) + beta * Out
	Tensor* matSgemm(Tensor* tA, Tensor* tB, float alpha, float beta,
		cublasOperation_t transA = CUBLAS_OP_N, cublasOperation_t transB = CUBLAS_OP_N);
	// tOut = alpha * ( tA x tB ) + beta * Out
	void matSgemm(Tensor* tOut, Tensor* tA, Tensor* tB, float alpha, float beta,
		cublasOperation_t transA = CUBLAS_OP_N, cublasOperation_t transB = CUBLAS_OP_N);

	// Return = tA x tB
	Tensor* matMult(Tensor* tA, Tensor* tB);
	// tOut = tA x tB
	void matMult(Tensor* tOut, Tensor* tA, Tensor* tB);

	// tB = (Scalar) scalA * (Tensor) tB
	void matMult(dtype scalA, Tensor* tB);

	// Return = alpha * tA + beta * tB
	Tensor* matSgeam(Tensor* tA, Tensor* tB, float alpha, float beta,
		cublasOperation_t transA = CUBLAS_OP_N, cublasOperation_t transB = CUBLAS_OP_N);
	// tOut = alpha * tA + beta * tB
	void matSgeam(Tensor* tOut, Tensor* tA, Tensor* tB, float alpha, float beta,
		cublasOperation_t transA = CUBLAS_OP_N, cublasOperation_t transB = CUBLAS_OP_N);

	// Return = tA + tB
	Tensor* matAdd(Tensor* tA, Tensor* tB);
	// tOut = tA + tB (Can be inplace)
	void matAdd(Tensor* tOut, Tensor* tA, Tensor* tB);

	// Return = tA - tB
	Tensor* matSub(Tensor* tA, Tensor* tB);
	// tOut = tA - tB  (Can be inplace)
	void matSub(Tensor* tOut, Tensor* tA, Tensor* tB);

	// Return = tA * tB (Haramard Product)
	Tensor* matEltMult(Tensor* tA, Tensor* tB);
	// tOut = tA * tB (Haramard Product, Can be inplace)
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

	// The other operations
	void getGpuInformation();

	void getCuDnnVersion();

	void getCuBlasVersion();

	void getGpuDriverVersion();

	int deviceId() {
		return mDeviceId;
	}

	// Allocate tensor t to device and return device pointer of allocated tensor
	void sendTensorToDevice(Tensor* t, bool sendData = true);

	// Allocate tensor t data to device and return device pointer of allocated data
	void sendDataToDevice(Tensor* t);

	// Retrieve tensor t data from tensor device pointer
	void retrievDataFromDevice(Tensor* t, bool retreiveOnlyData = true);

	// Fill tensor tA with value with cuDNN
	void fill(Tensor* tA, dtype value);

	void convolution(Tensor* tOutput, Tensor* tInput, Tensor* tFilter, int* strides, int* padding);

	void fullyConnected(Tensor* tOutput, Tensor* tInput, Tensor* tBias);

	void activation(Tensor* tOutput, Tensor* tInput, cudnnActivationMode_t activationMode);

	void operation(Tensor* tOutput, Tensor* tInput, Tensor* tInput2, cudnnOpTensorOp_t operation);

	void softmax(Tensor* tOutput, Tensor* tInput);

	void batchNormalization(Tensor* tOutput, Tensor* tInput);

	cudnnHandle_t cudnnHandle() {
		return mCudnnHandle;
	}

	cublasHandle_t cublasHandle() {
		return mCublasHandle;
	}

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
		sendTensorToDevice(&bias);
		sm.endTime(0, "Send bias");
		sm.startTime(0);
		sendTensorToDevice(&dest);
		sm.endTime(0, "Send dest");
		float value = 3.5;
		cudnnSetTensor(
			mCudnnHandle,
			t_desc,
			dest.devDataPtr(),
			&value
		);

		float alpha = 1;
		CuDNN_ERROR(
			cudnnAddTensor( // Add bias to dest
				mCudnnHandle,
				&alpha, // Coefficient
				t_desc, // Bias tensor descriptor
				bias.devDataPtr(), // Bias tensor data pointer
				&alpha, // Coefficient
				t_desc, // Destination tensor descriptor
				dest.devDataPtr() // Destination tensor data
			)
		);

		sendTensorToDevice(&dest2);
		cudnnOpTensorDescriptor_t t_op_desc;
		cudnnCreateOpTensorDescriptor(&t_op_desc);
		cudnnSetOpTensorDescriptor(t_op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);

		float beta = 0;
		CuDNN_ERROR(
			cudnnOpTensor(
				mCudnnHandle,
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
				mCudnnHandle,
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

		sm.startTime(0);
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
		sm.endTime(0, "Conv get dim");

		Tensor conv1({ n, c, h, w }, "Conv1", 0);
		conv1.printShape();
		sm.startTime(1);
		cudnnTensorDescriptor_t conv1_desc;
		cudnnCreateTensorDescriptor(&conv1_desc);
		cudnnSetTensor4dDescriptor(conv1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
		sm.endTime(1, "Descriptor create time");


		sm.startTime(0);
		sendTensorToDevice(&conv_filter1);
		matRand(&conv_filter1, -1, 1);
		sm.endTime(0, "Send filter");
		sm.startTime(0);
		sendTensorToDevice(&conv1);
		sm.endTime(0, "Send conv");
		sm.startTime(1);
		int returnedAlgoCount[3];
		cudnnConvolutionFwdAlgoPerf_t conv_perf[3];
		cudnnFindConvolutionForwardAlgorithm(
			mCudnnHandle,
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
		sendTensorToDevice(&conv_input);
		dtype k = 1;
		cudnnSetTensor(
			mCudnnHandle,
			t_desc,
			conv_input.devDataPtr(),
			&k
		);

		retrievDataFromDevice(&conv_filter1);
		conv_filter1.printTensor();

		retrievDataFromDevice(&conv_input);
		conv_input.printTensor();

		size_t convFwdWorkspaceSize;
		cudnnGetConvolutionForwardWorkspaceSize(
			mCudnnHandle,
			t_desc,
			filter_desc,
			conv_desc,
			t_desc,
			conv_perf[0].algo,
			&convFwdWorkspaceSize
		);

		dtype* convFwdWorkspace;
		cudaMalloc((void**)&convFwdWorkspace, convFwdWorkspaceSize);

		alpha = 1; beta = 0;
		CuDNN_ERROR(
			cudnnConvolutionForward( // Column major approach
				mCudnnHandle, // Handle
				&alpha, // Alpha : Input tensor coefficient
				t_desc, // xDesc
				conv_input.devDataPtr(), // *x
				filter_desc, // wDesc
				conv_filter1.devDataPtr(), // *w
				conv_desc, // convDesc
				CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, // Algo
				convFwdWorkspace, // *workSpace
				convFwdWorkspaceSize, // workSpaceSizeInBytes
				&beta, // *beta : Output tensor coefficient
				conv1_desc, // yDesc
				conv1.devDataPtr() // *y
			)
		);

		retrievDataFromDevice(&conv1);
		conv1.printTensor();

		int requestCount = 3;
		int returnCount;
		cudnnConvolutionBwdFilterAlgoPerf_t perfResult;
		cudnnFindConvolutionBackwardFilterAlgorithm(
			mCudnnHandle,
			t_desc,
			t_desc,
			conv_desc,
			filter_desc,
			requestCount,
			&returnCount,
			&perfResult
		);
		cout << "Convolution Backward Filter Algorithm : " << perfResult.algo << endl;
		cout << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 << endl;

		cudnnConvolutionBwdDataAlgoPerf_t perfData;
		cudnnFindConvolutionBackwardDataAlgorithm(
			mCudnnHandle,
			filter_desc,
			t_desc,
			conv_desc,
			t_desc,
			requestCount,
			&returnCount,
			&perfData
		);

		cout << "Convolution Bacward Data Algorithm : " << perfData.algo << endl;
		cout << CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD << endl;

		size_t backfilterWorkspaceSize;
		cudnnGetConvolutionBackwardFilterWorkspaceSize(
			mCudnnHandle,
			t_desc,
			t_desc,
			conv_desc,
			filter_desc,
			perfResult.algo,
			&backfilterWorkspaceSize
		);

		Tensor backfilterGradTensor({ 1, 1, 3, 3 }, "backFilterGradTensor", 0);
		sendTensorToDevice(&backfilterGradTensor);
		dtype* workspaceFilter;
		cudaMalloc((void**)&workspaceFilter, backfilterWorkspaceSize);
		cudnnConvolutionBackwardFilter(
			mCudnnHandle,
			&alpha,
			t_desc,
			conv_input.devDataPtr(),
			t_desc,
			conv1.devDataPtr(),
			conv_desc,
			perfResult.algo,
			workspaceFilter,
			backfilterWorkspaceSize,
			&beta,
			filter_desc,
			backfilterGradTensor.devDataPtr()
		);
		retrievDataFromDevice(&backfilterGradTensor);
		backfilterGradTensor.printTensor();


		size_t backWorkspaceSize;
		cudnnGetConvolutionBackwardDataWorkspaceSize(
			mCudnnHandle,
			filter_desc,
			t_desc,
			conv_desc,
			t_desc,
			CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
			&backWorkspaceSize
		);
		cout << "Backward Workspace Size : " << backWorkspaceSize << endl;

		Tensor backGradTensor({ 1, 1, 3, 3 }, "backGradTensor", 0);
		sendTensorToDevice(&backGradTensor);
		dtype* workspace;
		cudaMalloc((void**)&workspace, backWorkspaceSize);
		cudnnConvolutionBackwardData(
			mCudnnHandle,
			&alpha,
			filter_desc,
			conv_filter1.devDataPtr(),
			t_desc,
			conv1.devDataPtr(),
			conv_desc,
			CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
			workspace,
			backWorkspaceSize,
			&beta,
			t_desc,
			backGradTensor.devDataPtr()
		);
		retrievDataFromDevice(&backGradTensor);
		backGradTensor.printTensor();


		Tensor softmaxOut({ 1, 1, 3, 3 });
		sendTensorToDevice(&softmaxOut);
		cudnnSoftmaxForward(
			mCudnnHandle,
			CUDNN_SOFTMAX_FAST,
			CUDNN_SOFTMAX_MODE_INSTANCE,
			&alpha,
			t_desc,
			conv1.devDataPtr(),
			&beta,
			t_desc,
			softmaxOut.devDataPtr()
		);
		retrievDataFromDevice(&softmaxOut);
		retrievDataFromDevice(&conv1);
		conv1.printTensor();
		softmaxOut.printTensor();

		Tensor softmaxGrad({ 1, 1, 3, 3 }, "softmaxGrad", 0);
		sendTensorToDevice(&softmaxGrad);
		CuDNN_ERROR(
			cudnnSoftmaxBackward(
				mCudnnHandle,
				CUDNN_SOFTMAX_FAST,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&alpha,
				t_desc,
				softmaxOut.devDataPtr(),
				t_desc,
				softmaxGrad.devDataPtr(),
				&beta,
				t_desc,
				softmaxGrad.devDataPtr()
			)
		);
		retrievDataFromDevice(&softmaxGrad);
		softmaxGrad.printTensor();

		cudnnPoolingDescriptor_t pooling_desc;
		cudnnCreatePoolingDescriptor(&pooling_desc);
		cudnnSetPooling2dDescriptor(
			pooling_desc,
			CUDNN_POOLING_MAX,
			CUDNN_NOT_PROPAGATE_NAN,
			2, // Window Height
			2, // Window Width
			0, // Vertical Padding
			0, // Horizontal Padding,
			2, // Vertical Stride
			2 // Horizontal stride
		);

		int poolOutN, poolOutC, poolOutW, poolOutH;
		cudnnGetPooling2dForwardOutputDim(
			pooling_desc,
			t_desc,
			&poolOutN,
			&poolOutC,
			&poolOutH,
			&poolOutW
		);

		cudnnTensorDescriptor_t pool_out_desc;
		cudnnCreateTensorDescriptor(&pool_out_desc);
		cudnnSetTensor4dDescriptor(
			pool_out_desc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			poolOutN,
			poolOutC,
			poolOutH,
			poolOutW
		);

		Tensor poolOutTensor({ poolOutN, poolOutC, poolOutH, poolOutW }, "poolOutTensor", 0);
		sendTensorToDevice(&poolOutTensor);

		cudnnPoolingForward(
			mCudnnHandle,
			pooling_desc,
			&alpha,
			t_desc,
			conv1.devDataPtr(),
			&beta,
			pool_out_desc,
			poolOutTensor.devDataPtr()
		);

		retrievDataFromDevice(&poolOutTensor);
		poolOutTensor.printTensor();


		Tensor poolBackTensor({ 1, 1, 3, 3 }, "poolBackTensor", 0);
		sendTensorToDevice(&poolBackTensor);
		cudnnPoolingBackward(
			mCudnnHandle,
			pooling_desc,
			&alpha,
			pool_out_desc,
			poolOutTensor.devDataPtr(),
			pool_out_desc,
			poolOutTensor.devDataPtr(),
			t_desc,
			conv1.devDataPtr(),
			&beta,
			t_desc,
			poolBackTensor.devDataPtr()
		);

		retrievDataFromDevice(&poolBackTensor);
		poolBackTensor.printTensor();

		cudnnActivationDescriptor_t activation_desc;
		cudnnCreateActivationDescriptor(&activation_desc);
		cudnnSetActivationDescriptor(
			activation_desc,
			CUDNN_ACTIVATION_RELU,
			CUDNN_NOT_PROPAGATE_NAN,
			0
		);

		Tensor reluOut({ 1, 1, 3, 3 }, "reluOut", 0);
		sendTensorToDevice(&reluOut);

		cudnnActivationForward(
			mCudnnHandle,
			activation_desc,
			&alpha,
			t_desc,
			conv1.devDataPtr(),
			&beta,
			t_desc,
			reluOut.devDataPtr()
		);

		retrievDataFromDevice(&reluOut);
		reluOut.printTensor();

		Tensor reluBack({ 1, 1, 3, 3 }, "reluBack", 0);
		sendTensorToDevice(&reluBack);

		cudnnActivationBackward(
			mCudnnHandle,
			activation_desc,
			&alpha,
			t_desc,
			reluOut.devDataPtr(),
			t_desc,
			reluOut.devDataPtr(),
			t_desc,
			conv1.devDataPtr(),
			&beta,
			t_desc,
			reluBack.devDataPtr()
		);
		retrievDataFromDevice(&reluBack);
		reluBack.printTensor();

		cudnnLRNDescriptor_t lrn_desc;
		cudnnCreateLRNDescriptor(&lrn_desc);
		cudnnSetLRNDescriptor(
			lrn_desc,
			1, // lrnN
			0.0001, // lrnAlpha
			0.75, // lrnBeta
			2.0 // lrnK
		);

		Tensor lrnOut({ 1, 1, 3, 3 }, "lrnOut", 0);
		sendTensorToDevice(&lrnOut);

		cudnnLRNCrossChannelForward(
			mCudnnHandle,
			lrn_desc,
			CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&alpha,
			t_desc,
			conv1.devDataPtr(),
			&beta,
			t_desc,
			lrnOut.devDataPtr()
		);
		retrievDataFromDevice(&lrnOut);
		lrnOut.printTensor();

		Tensor lrnBack({ 1, 1, 3, 3 }, "lrnBack", 0);
		sendTensorToDevice(&lrnBack);
		cudnnLRNCrossChannelBackward(
			mCudnnHandle,
			lrn_desc,
			CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&alpha,
			t_desc,
			lrnOut.devDataPtr(),
			t_desc,
			lrnOut.devDataPtr(),
			t_desc,
			conv1.devDataPtr(),
			&beta,
			t_desc,
			lrnBack.devDataPtr()
		);
		retrievDataFromDevice(&lrnBack);
		lrnBack.printTensor();

		//cudnnDivisiveNormalizationForward(
		//	cuDnnHandle,
		//	lrn_desc,
		//	CUDNN_DIVNORM_PRECOMPUTED_MEANS,
		//	&alpha,
		//	t_desc,
		//	conv1.devDataPtr(),

		//)

		Tensor batchNormOut({ 1, 1, 3, 3 }, "batchNormOut", 0);
		sendTensorToDevice(&batchNormOut);

		cudnnTensorDescriptor_t bnScaleBiasMeanDesc;
		cudnnCreateTensorDescriptor(&bnScaleBiasMeanDesc);
		cudnnSetTensor4dDescriptor(
			bnScaleBiasMeanDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1,
			1,
			3,
			3
		);

		dtype* bnScale, *bnBias;
		cudaMalloc((void**)&bnScale, sizeof(dtype) * 1 * 1 * 3 * 3);
		cudaMalloc((void**)&bnBias, sizeof(dtype) * 1 * 1 * 3 * 3);

		dtype* estimatedMean, *estimatedVariance;
		cudaMalloc((void**)&estimatedMean, sizeof(dtype) * 1 * 1 * 3 * 3);
		cudaMalloc((void**)&estimatedVariance, sizeof(dtype) * 1 * 1 * 3 * 3);

		cudnnBatchNormalizationForwardInference(
			mCudnnHandle,
			CUDNN_BATCHNORM_PER_ACTIVATION,
			&alpha,
			&beta,
			t_desc,
			conv1.devDataPtr(),
			t_desc,
			batchNormOut.devDataPtr(),
			bnScaleBiasMeanDesc,
			bnScale,
			bnBias,
			estimatedMean,
			estimatedVariance,
			CUDNN_BN_MIN_EPSILON
		);

		dtype* resultLearningMean, *resultRunningVaraiance;
		cudaMalloc((void**)&resultLearningMean, sizeof(dtype) * 1 * 1 * 3 * 3);
		cudaMalloc((void**)&resultRunningVaraiance, sizeof(dtype) * 1 * 1 * 3 * 3);

		dtype* resultSaveMean, *resultSaveInvVaraiance;
		cudaMalloc((void**)&resultSaveMean, sizeof(dtype) * 1 * 1 * 3 * 3);
		cudaMalloc((void**)&resultSaveInvVaraiance, sizeof(dtype) * 1 * 1 * 3 * 3);

		cudnnBatchNormalizationForwardTraining(
			mCudnnHandle,
			CUDNN_BATCHNORM_PER_ACTIVATION,
			&alpha,
			&beta,
			t_desc,
			conv1.devDataPtr(),
			t_desc,
			batchNormOut.devDataPtr(),
			bnScaleBiasMeanDesc,
			bnScale,
			bnBias,
			0.5,
			resultLearningMean,
			resultRunningVaraiance,
			CUDNN_BN_MIN_EPSILON,
			resultSaveMean,
			resultSaveInvVaraiance
		);

		dtype* dBnScaleResult, *dBnBiasResult;
		cudaMalloc((void**)&dBnScaleResult, sizeof(dtype) * 1 * 1 * 3 * 3);
		cudaMalloc((void**)&dBnBiasResult, sizeof(dtype) * 1 * 1 * 3 * 3);

		dtype* savedMean, *savedInvVariance;
		cudaMalloc((void**)&savedMean, sizeof(dtype) * 1 * 1 * 3 * 3);
		cudaMalloc((void**)&savedInvVariance, sizeof(dtype) * 1 * 1 * 3 * 3);

		cudnnBatchNormalizationBackward(
			mCudnnHandle,
			CUDNN_BATCHNORM_PER_ACTIVATION,
			&alpha, &beta, &alpha, &beta,
			t_desc,
			conv1.devDataPtr(),
			t_desc,
			batchNormOut.devDataPtr(),
			t_desc,
			conv1.devDataPtr(),
			bnScaleBiasMeanDesc,
			bnScale,
			dBnScaleResult,
			dBnBiasResult,
			CUDNN_BN_MIN_EPSILON,
			savedMean,
			savedInvVariance
		);

		cudnnDropoutDescriptor_t dropout_desc;
		cudnnCreateDropoutDescriptor(&dropout_desc);
		
		size_t dropoutStatesSize;
		cudnnDropoutGetStatesSize(
			mCudnnHandle,
			&dropoutStatesSize
		);

		size_t dropoutReserveSpaceSize;
		cudnnDropoutGetReserveSpaceSize(
			t_desc,
			&dropoutReserveSpaceSize
		);
		
		dtype* dropout_states;
		cudaMalloc((void**)&dropout_states, dropoutStatesSize);
		cudnnSetDropoutDescriptor(
			dropout_desc,
			mCudnnHandle,
			0.5, // Dropout rate
			dropout_states,
			dropoutStatesSize,
			11111 // Seed
		);


		Tensor dropOutFwd({ 1, 1, 3, 3 }, "dropOutFwd", 0);
		sendTensorToDevice(&dropOutFwd);

		dtype* reserved_space;
		cudaMalloc((void**)&reserved_space, dropoutReserveSpaceSize);

		Tensor dropSample({ 1, 1, 3, 3 }, "dropSample", 0);
		for (int i = 0; i < dropSample.col(); i++)
			for (int j = 0; j < dropSample.row(); j++)
				dropSample(0, 0, i, j) = i + j * dropSample.col();

		sendTensorToDevice(&dropSample);
		dropSample.printTensor();

		cudnnDropoutForward(
			mCudnnHandle,
			dropout_desc,
			t_desc,
			dropSample.devDataPtr(),
			t_desc,
			dropOutFwd.devDataPtr(),
			reserved_space,
			dropoutReserveSpaceSize
		);

		retrievDataFromDevice(&dropOutFwd);
		dropOutFwd.printTensor();


		Tensor dropBack({ 1, 1, 3, 3 }, "dropBack", 0);
		sendTensorToDevice(&dropBack);

		cudnnDropoutBackward(
			mCudnnHandle,
			dropout_desc,
			t_desc,
			dropOutFwd.devDataPtr(),
			t_desc,
			dropBack.devDataPtr(),
			reserved_space,
			dropoutReserveSpaceSize
		);

		retrievDataFromDevice(&dropBack);
		dropBack.printTensor();


		Tensor sampler({ 1, 1, 10, 10 }, "sampler", 0);
		for (int i = 0; i < sampler.col(); i++)
			for (int j = 0; j < sampler.row(); j++)
				sampler(0, 0, i, j) = i + j * sampler.col();

		sendTensorToDevice(&sampler);
		sampler.printTensor();

		cudnnSpatialTransformerDescriptor_t trans_desc;
		cudnnCreateSpatialTransformerDescriptor(&trans_desc);
		cudnnSetSpatialTransformerNdDescriptor(
			trans_desc,
			CUDNN_SAMPLER_BILINEAR,
			CUDNN_DATA_FLOAT,
			sampler.dimension(),
			sampler.mShape
		);

		dtype* theta; // Affine transformation matrix
		cudaMalloc((void**)&theta, sizeof(dtype) * sampler.shape(0) * 2 * 3);

		Tensor grid({ sampler.n(), sampler.h(), sampler.w(), 2 }, "grid", 0);
		sendTensorToDevice(&grid);

		cudnnSpatialTfGridGeneratorForward(
			mCudnnHandle,
			trans_desc,
			theta,
			grid.devDataPtr()
		);

		retrievDataFromDevice(&grid);
		grid.printTensor();

		//cudnnSpatialTfGridGeneratorBackward();
		//cudnnSpatialTfSamplerBackward();

		//cudnnRNNDescriptor_t rnn_desc;
		//cudnnCreateRNNDescriptor(&rnn_desc);
		//cudnnSetRNNDescriptor(
		//	rnn_desc,
		//	1024,
		//	2,
		//	dropout_desc,
		//	CUDNN_LINEAR_INPUT,
		//	CUDNN_UNIDIRECTIONAL,
		//	CUDNN_RNN_RELU,
		//	CUDNN_DATA_FLOAT
		//);

		//cudnnTensorDescriptor_t rnn_workspace_desc;
		//cudnnCreateTensorDescriptor(&rnn_workspace_desc);
		//size_t rnn_workspace_size;

		//cudnnGetRNNWorkspaceSize(
		//	cuDnnHandle,
		//	rnn_desc,
		//	1, // Seq Length
		//	&rnn_workspace_desc,
		//	&rnn_workspace_size
		//);

		//cudnnTensorDescriptor_t rnn_training_reverse_desc;
		//cudnnCreateTensorDescriptor(&rnn_training_reverse_desc);
		//size_t rnn_training_reverse_size;

		//cudnnGetRNNTrainingReserveSize(
		//	cuDnnHandle,
		//	rnn_desc,
		//	1, // Seq Length
		//	&rnn_training_reverse_desc,
		//	&rnn_training_reverse_size
		//);

		//cudnnRNNForwardInference(
		//	cuDnnHandle,
		//	rnn_desc,
		//	1,
		//	&
		//);
		//cudnnRNNForwardTraining(

		//);
		//cudnnRNNBackwardData(

		//);
		//cudnnRNNBackwardWeights(

		//);
		//cudnnCTCLoss();


		cudnnDestroyTensorDescriptor(t_desc);
		cudnnDestroyOpTensorDescriptor(t_op_desc);
		cudnnDestroyFilterDescriptor(filter_desc);
		cudnnDestroyConvolutionDescriptor(conv_desc);
		cudnnDestroyPoolingDescriptor(pooling_desc);
		cudnnDestroyActivationDescriptor(activation_desc);
		cudnnDestroyLRNDescriptor(lrn_desc);
		//cudnnDestroyRNNDescriptor(rnn_desc);
		cudnnDestroyDropoutDescriptor(dropout_desc);
		cudnnDestroySpatialTransformerDescriptor(trans_desc);
	}
};

#endif