#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"

class ActivationLayer : Layer {
private:
	// Descriptor for activation layer
	cudnnActivationDescriptor_t mActDesc;

	// Activation mode
	cudnnActivationMode_t mActMode;

	cudnnNanPropagation_t mPropagateNan;

	dtype mAlpha;
	dtype mBeta;

	Tensor* mInput;
	Tensor* mOutput;

public:
	ActivationLayer(Perceptor* aPerceptor, cudnnActivationMode_t aActMode,
		cudnnNanPropagation_t aPropagteNan = CUDNN_NOT_PROPAGATE_NAN, dtype aReluCeiling = 0) {

		mPerceptor = aPerceptor;
		mActMode = aActMode;
		mPropagateNan = aPropagteNan;

		mAlpha = 1;
		mBeta = 0;

		CuDNN_ERROR(cudnnCreateActivationDescriptor(&mActDesc));
		CuDNN_ERROR(
			cudnnSetActivationDescriptor(
				mActDesc,
				mActMode,
				mPropagateNan,
				aReluCeiling
			)
		);
	}

	~ActivationLayer() {

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