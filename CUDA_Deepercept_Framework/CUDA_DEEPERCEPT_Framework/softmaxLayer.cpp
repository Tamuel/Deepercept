#include "softmaxLayer.h"


// tInput -> Propagate -> tOutput
void SoftmaxLayer::forwardPropagation(Tensor* tInput, Tensor* tOutput) {
	CuDNN_ERROR(
		cudnnSoftmaxForward(
			mPerceptor->cudnnHandle(),
			CUDNN_SOFTMAX_FAST,
			CUDNN_SOFTMAX_MODE_INSTANCE,
			&mAlpha,
			tInput->descriptor(),
			tInput->devDataPtr(),
			&mBeta,
			tOutput->descriptor(),
			tOutput->devDataPtr()
		)
	);

	mInput = tInput;
	mOutput = tOutput;
}

// gOutput <- Backpropagate <- gInput
void SoftmaxLayer::backwardPropagation(Tensor* gInput, Tensor* gOutput) {
	CuDNN_ERROR(
		cudnnSoftmaxBackward(
			mPerceptor->cudnnHandle(),
			CUDNN_SOFTMAX_FAST,
			CUDNN_SOFTMAX_MODE_INSTANCE,
			&mAlpha,
			mOutput->descriptor(),
			mOutput->devDataPtr(),
			gInput->descriptor(),
			gInput->devDataPtr(),
			&mBeta,
			gOutput->descriptor(),
			gOutput->devDataPtr()
		)
	);
}

// tInput -> Propagate -> tOutput
void SoftmaxLayer::forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) {

}

// gOutput <- Backpropagate <- gInput
void SoftmaxLayer::backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) {

}