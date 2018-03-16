#ifndef OP_LAYER_H
#define OP_LAYER_H

#include "layer.h"

class SoftmaxLayer : Layer {
private:
	dtype mAlpha;
	dtype mBeta;

	Tensor* mInput;
	Tensor* mOutput;

public:
	SoftmaxLayer(Perceptor* aPerceptor) {
		mPerceptor = aPerceptor;

		mAlpha = 1;
		mBeta = 0;

	}

	~SoftmaxLayer() {

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