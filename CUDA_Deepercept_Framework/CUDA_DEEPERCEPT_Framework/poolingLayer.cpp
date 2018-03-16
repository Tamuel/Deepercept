#include "poolingLayer.h"

// tInput -> Propagate -> tOutput
void PoolingLayer::forwardPropagation(Tensor* tInput, Tensor* tOutput) {
	CuDNN_ERROR(
		cudnnGetPooling2dForwardOutputDim(
			mPoolingDesc,
			tInput->descriptor(),
			&mPoolOutN,
			&mPoolOutC,
			&mPoolOutH,
			&mPoolOutW
		)
	);
	CuDNN_ERROR(
		cudnnPoolingForward(
			mPerceptor->cudnnHandle(),
			mPoolingDesc,
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
void PoolingLayer::backwardPropagation(Tensor* gInput, Tensor* gOutput) {
	CuDNN_ERROR(
		cudnnPoolingBackward(
			mPerceptor->cudnnHandle(),
			mPoolingDesc,
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
void PoolingLayer::forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) {

}

// gOutput <- Backpropagate <- gInput
void PoolingLayer::backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) {

}