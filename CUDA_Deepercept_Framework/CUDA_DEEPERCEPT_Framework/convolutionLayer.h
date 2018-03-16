#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"

class ConvolutionLayer : Layer {
private:
	// Descriptor for convolution layer
	cudnnConvolutionDescriptor_t mConvDesc;

	cudnnFilterDescriptor_t mConvFilterDesc;

	cudnnConvolutionMode_t mConvMode;

	cudnnConvolutionFwdAlgo_t mForwardAlg;

	cudnnConvolutionBwdFilterAlgo_t mFilterBackwardAlg;

	cudnnConvolutionBwdDataAlgo_t mDataBackwardAlg;

	cudnnDataType_t mDatatype;

	size_t mForwardWorkspaceSize;
	dtype* mForwardWorkspace;

	size_t mFilterBackwardWorkspaceSize;
	dtype* mFilterBackwardWorkspace;

	size_t mDataBackwardWorkspaceSize;
	dtype* mDataBackwardWorkspace;

	int mVpadding;
	int mHpadding;
	int mVstride;
	int mHstride;
	int mVdialation;
	int mHdialation;

	int mConvOutN;
	int mConvOutC;
	int mConvOutH;
	int mConvOutW;

	dtype mAlpha;
	dtype mBeta;

	Tensor* mInput;
	Tensor* mFilter;
	Tensor* mOutput;

	Tensor* mFilterGrad;

	bool mFilterIsInitilized;
	bool mFwdAlgAndWorkspaceIsInitilized;
	bool mBakAlgAndWorkspaceIsInitilized;

	dtype mLearningRate;

public:
	ConvolutionLayer(Perceptor* aPerceptor,
		int aVpadding, int aHpadding, int aVstride, int aHstride, int aVdialation, int aHdialation,
		Tensor* aFilter,
		cudnnDataType_t aDatatype, cudnnConvolutionMode_t aConvMode = CUDNN_CROSS_CORRELATION)
		: mFilterIsInitilized(false), mFwdAlgAndWorkspaceIsInitilized(false), mBakAlgAndWorkspaceIsInitilized(false), mLearningRate(0.01) {

		mPerceptor = aPerceptor;
		setFilter(aFilter);
		
		mAlpha = 1;
		mBeta = 0;

		CuDNN_ERROR(cudnnCreateConvolutionDescriptor(&mConvDesc));
		CuDNN_ERROR(
			cudnnSetConvolution2dDescriptor(
				mConvDesc,
				aVpadding,
				aHpadding,
				aVstride,
				aHstride,
				aVdialation,
				aHdialation,
				aConvMode
			)
		);
		mConvMode = aConvMode;
		mDatatype = aDatatype;

		mVpadding = aVpadding;
		mHpadding = aHpadding;
		mVstride = aVstride;
		mHstride = aHstride;
		mVdialation = aVdialation;
		mHdialation = aHdialation;
	}

	~ConvolutionLayer() {
		CuDNN_ERROR(cudnnDestroyConvolutionDescriptor(mConvDesc));
		cudaFree(mForwardWorkspace);
		cudaFree(mFilterBackwardWorkspace);
		delete mFilterGrad;
	}

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInput, Tensor* tOutput);

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutput);

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput);

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB);

	void setFilter(Tensor* tFilter);

	void setForwardAlgorithmAndWorkspace(Tensor* tInput, Tensor* tOutput);

	void setBackwardAlgorithmAndWorkspace(Tensor* gInput, Tensor* gOutput);

	void update(Tensor* tFilter, Tensor* tGrad, dtype learningRate);
};

#endif