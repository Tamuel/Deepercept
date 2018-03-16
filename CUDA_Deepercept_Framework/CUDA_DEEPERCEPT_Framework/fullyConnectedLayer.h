#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"

class FullyConnectedLayer : Layer {
private:

	Tensor* mNeurons;
	Tensor* mNeuronsGrad;

	Tensor* mInput;
	Tensor* mOutput;

	dtype mAlpha;
	dtype mBeta;

	dtype mLearningRate;

public:
	FullyConnectedLayer(Perceptor* aPerceptor, Tensor* aPerceptrons) {
		mPerceptor = aPerceptor;
		mNeurons = aPerceptrons;

		setGradient(mNeuronsGrad);

		mAlpha = 1;
		mBeta = 0;
	}

	~FullyConnectedLayer() {
		delete mNeuronsGrad;
	}

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInput, Tensor* tOutput);

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutput);

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput);

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB);

	void setGradient(Tensor* aPerceptrons);

	void update(Tensor* tNeurons, Tensor* tGrad, dtype learningRate);
};

#endif