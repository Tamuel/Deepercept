#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include "layer.h"

class BatchNormalizationLayer : Layer {
private:

public:
	BatchNormalizationLayer(Perceptor* aPerceptor) {

	}
	
	~BatchNormalizationLayer() {

	}

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInput, Tensor* tOutput) = 0;

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutput) = 0;

	// tInput -> Propagate -> tOutput
	virtual void forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) = 0;

	// gOutput <- Backpropagate <- gInput
	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) = 0;
};

#endif