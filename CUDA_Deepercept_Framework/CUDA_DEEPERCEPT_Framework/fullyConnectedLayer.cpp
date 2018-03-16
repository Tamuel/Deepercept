#include "fullyConnectedLayer.h"


// tInput -> Propagate -> tOutput, tInput and tOutput must be vector
void FullyConnectedLayer::forwardPropagation(Tensor* tInput, Tensor* tOutput) {
	mPerceptor->matSgemm(tOutput, tInput, mNeurons, mAlpha, mBeta, CUBLAS_OP_T, CUBLAS_OP_N);
	mInput = tInput;
	mOutput = tOutput;
}

// gOutput <- Backpropagate <- gInput
void FullyConnectedLayer::backwardPropagation(Tensor* gInput, Tensor* gOutput) {
	// Data backward propagation
	mPerceptor->matSgemm(gOutput, mNeurons, gInput, mAlpha, mBeta);

	// Neuron backward propagation
	mPerceptor->matSgemm(mNeuronsGrad, mInput, gInput, mAlpha, mBeta, CUBLAS_OP_N, CUBLAS_OP_T);
}

// tInput -> Propagate -> tOutput
void FullyConnectedLayer::forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) {

}

// gOutput <- Backpropagate <- gInput
void FullyConnectedLayer::backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) {

}

void FullyConnectedLayer::setGradient(Tensor* aPerceptrons) {
	mNeuronsGrad = new Tensor({ aPerceptrons->n(), aPerceptrons->c(), aPerceptrons->h(), aPerceptrons->w() });
}

void FullyConnectedLayer::update(Tensor* tNeurons, Tensor* tGrad, dtype learningRate) {
	mPerceptor->matSgeam(tNeurons, tNeurons, tGrad, 1, learningRate);
}