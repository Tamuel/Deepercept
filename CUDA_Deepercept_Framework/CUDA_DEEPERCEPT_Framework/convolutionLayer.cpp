#include "convolutionLayer.h"

// tInput -> Propagate -> tOutput
void ConvolutionLayer::forwardPropagation(Tensor* tInput, Tensor* tOutput) {
	if (!mFilterIsInitilized) setFilter(mFilter);
	if (!mFwdAlgAndWorkspaceIsInitilized) setForwardAlgorithmAndWorkspace(tInput, tOutput);

	CuDNN_ERROR(
		cudnnConvolutionForward(
			mPerceptor->cudnnHandle(),
			&mAlpha,
			tInput->descriptor(),
			tInput->devDataPtr(),
			mConvFilterDesc,
			mFilter->devDataPtr(),
			mConvDesc,
			mForwardAlg,
			mForwardWorkspace,
			mForwardWorkspaceSize,
			&mBeta,
			tOutput->descriptor(),
			tOutput->devDataPtr()
		)
	);
}

// gOutput <- Backpropagate <- gInput
void ConvolutionLayer::backwardPropagation(Tensor* gInput, Tensor* gOutput) {
	if (!mBakAlgAndWorkspaceIsInitilized) setBackwardAlgorithmAndWorkspace(gInput, gOutput);
	CuDNN_ERROR(
		cudnnConvolutionBackwardFilter(
			mPerceptor->cudnnHandle(),
			&mAlpha,
			mInput->descriptor(),
			mInput->devDataPtr(),
			gInput->descriptor(),
			gInput->devDataPtr(),
			mConvDesc,
			mFilterBackwardAlg,
			mFilterBackwardWorkspace,
			mFilterBackwardWorkspaceSize,
			&mBeta,
			mConvFilterDesc,
			mFilterGrad->devDataPtr()
		)
	);

	CuDNN_ERROR(
		cudnnConvolutionBackwardData(
			mPerceptor->cudnnHandle(),
			&mAlpha,
			mConvFilterDesc,
			mFilter->devDataPtr(),
			gInput->descriptor(),
			gInput->devDataPtr(),
			mConvDesc,
			mDataBackwardAlg,
			mDataBackwardWorkspace,
			mDataBackwardWorkspaceSize,
			&mBeta,
			gOutput->descriptor(),
			gOutput->devDataPtr()
		)
	);
}

// tInput -> Propagate -> tOutput
void ConvolutionLayer::forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) {

}

// gOutput <- Backpropagate <- gInput
void ConvolutionLayer::backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) {

}

void ConvolutionLayer::setFilter(Tensor* tFilter) {
	CuDNN_ERROR(cudnnCreateFilterDescriptor(&mConvFilterDesc));
	CuDNN_ERROR(
		cudnnSetFilter4dDescriptor(
			mConvFilterDesc,
			mDatatype,
			CUDNN_TENSOR_NCHW,
			tFilter->n(),
			tFilter->c(),
			tFilter->h(),
			tFilter->w()
		)
	);
	if (mFilterGrad) delete mFilterGrad;
	mFilterGrad = new Tensor({ tFilter->n(), tFilter->c(), tFilter->h(), tFilter->w() });
	mPerceptor->sendTensorToDevice(mFilterGrad);
	mFilter = tFilter;
	mFilterIsInitilized = true;
}

void ConvolutionLayer::setForwardAlgorithmAndWorkspace(Tensor* tInput, Tensor* tOutput) {
	// Find efficient algorithm for forward convolution
	cudnnConvolutionFwdAlgoPerf_t convPerf;
	int returnedAlgoCount;
	CuDNN_ERROR(
		cudnnFindConvolutionForwardAlgorithm(
			mPerceptor->cudnnHandle(),
			tInput->descriptor(),
			mConvFilterDesc,
			mConvDesc,
			tOutput->descriptor(),
			1,
			&returnedAlgoCount,
			&convPerf
		)
	);
	mForwardAlg = convPerf.algo;

	// Find workspace
	CuDNN_ERROR(
		cudnnGetConvolutionForwardWorkspaceSize(
			mPerceptor->cudnnHandle(),
			tInput->descriptor(),
			mConvFilterDesc,
			mConvDesc,
			tOutput->descriptor(),
			mForwardAlg,
			&mForwardWorkspaceSize
		)
	);
	cudaMalloc((void**)&mForwardWorkspace, mForwardWorkspaceSize);

	mInput = tInput;
	mOutput = tOutput;

	mFwdAlgAndWorkspaceIsInitilized = true;
}

void ConvolutionLayer::setBackwardAlgorithmAndWorkspace(Tensor* gInput, Tensor* gOutput) {
	// Find efficient algorithm for filter backward convolution
	cudnnConvolutionBwdFilterAlgoPerf_t convFilterPerf;
	int returnedAlgoCount;
	CuDNN_ERROR(
		cudnnFindConvolutionBackwardFilterAlgorithm(
			mPerceptor->cudnnHandle(),
			mInput->descriptor(),
			gInput->descriptor(),
			mConvDesc,
			mConvFilterDesc,
			1,
			&returnedAlgoCount,
			&convFilterPerf
		)
	);
	mFilterBackwardAlg = convFilterPerf.algo;

	// Find workspace
	CuDNN_ERROR(
		cudnnGetConvolutionBackwardFilterWorkspaceSize(
			mPerceptor->cudnnHandle(),
			mInput->descriptor(),
			gInput->descriptor(),
			mConvDesc,
			mConvFilterDesc,
			mFilterBackwardAlg,
			&mFilterBackwardWorkspaceSize
		)
	);
	cudaMalloc((void**)&mFilterBackwardWorkspace, mFilterBackwardWorkspaceSize);

	// Find efficient algorithm for data backward convolution
	cudnnConvolutionBwdDataAlgoPerf_t convDataPerf;
	CuDNN_ERROR(
		cudnnFindConvolutionBackwardDataAlgorithm(
			mPerceptor->cudnnHandle(),
			mConvFilterDesc,
			gInput->descriptor(),
			mConvDesc,
			gOutput->descriptor(),
			1,
			&returnedAlgoCount,
			&convDataPerf
		)
	);
	mDataBackwardAlg = convDataPerf.algo;

	// Find workspace
	CuDNN_ERROR(
		cudnnGetConvolutionBackwardDataWorkspaceSize(
			mPerceptor->cudnnHandle(),
			mConvFilterDesc,
			gInput->descriptor(),
			mConvDesc,
			gOutput->descriptor(),
			mDataBackwardAlg,
			&mDataBackwardWorkspaceSize
		)
	);
	cudaMalloc((void**)&mDataBackwardWorkspaceSize, mDataBackwardWorkspaceSize);

	mBakAlgAndWorkspaceIsInitilized = true;
}

void ConvolutionLayer::update(Tensor* tFilter, Tensor* tGrad, dtype learningRate) {
	mPerceptor->matSgeam(tFilter, tFilter, tGrad, 1, learningRate);
}