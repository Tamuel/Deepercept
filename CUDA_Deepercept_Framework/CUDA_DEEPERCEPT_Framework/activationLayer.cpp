#include "activationLayer.h"


// tInput -> Propagate -> tOutput
void ActivationLayer::forwardPropagation(Tensor* tInput, Tensor* tOutput) {
	CuDNN_ERROR(
		cudnnActivationForward(
			mPerceptor->cudnnHandle(),
			mActDesc,
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
void ActivationLayer::backwardPropagation(Tensor* gInput, Tensor* gOutput) {
	CuDNN_ERROR(
		cudnnActivationBackward(
			mPerceptor->cudnnHandle(),
			mActDesc,
			&mAlpha,
			mOutput->descriptor(),
			mOutput->devDataPtr(),
			gInput->descriptor(),
			gInput->devDataPtr(),
			mInput->descriptor(),
			mInput->devDataPtr(),
			&mBeta,
			gOutput->descriptor(),
			gOutput->devDataPtr()
		)
	);
}

// tInput -> Propagate -> tOutput
void ActivationLayer::forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) {

}

// gOutput <- Backpropagate <- gInput
void ActivationLayer::backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) {

}