#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"

class PoolingLayer : Layer {
private:
	// Descriptor for pooling layer
	cudnnPoolingDescriptor_t mPoolingDesc;

	int mWindowHeight;
	int mWindowWidth;
	int mVpadding;
	int mHpadding;
	int mVstride;
	int mHstride;

	int mPoolOutN;
	int mPoolOutC;
	int mPoolOutH;
	int mPoolOutW;

	dtype mAlpha;
	dtype mBeta;

	Tensor* mInput;
	Tensor* mOutput;

public:
	PoolingLayer(Perceptor* aPerceptor, cudnnPoolingMode_t aPoolingMode,
		int aWindowHeight, int aWindowWidth, int aVpadding, int aHpadding,
		int aVstride, int aHstride, cudnnNanPropagation_t aPropagateNan = CUDNN_NOT_PROPAGATE_NAN) {

		mPerceptor = aPerceptor;

		mAlpha = 1;
		mBeta = 0;

		CuDNN_ERROR(cudnnCreatePoolingDescriptor(&mPoolingDesc));
		CuDNN_ERROR(
			cudnnSetPooling2dDescriptor(
				mPoolingDesc,
				aPoolingMode,
				aPropagateNan,
				aWindowHeight,
				aWindowWidth,
				aVpadding,
				aHpadding,
				aVstride,
				aHstride
			)
		);

		mWindowHeight = aWindowHeight;
		mWindowWidth = aWindowWidth;
		mVpadding = aVpadding;
		mHpadding = aHpadding;
		mVstride = aVstride;
		mHstride = aHstride;

	}

	~PoolingLayer() {
		CuDNN_ERROR(cudnnDestroyPoolingDescriptor(mPoolingDesc));
	}

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInput, Tensor* tOutput);

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutput);

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput);

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB);
};

#endif