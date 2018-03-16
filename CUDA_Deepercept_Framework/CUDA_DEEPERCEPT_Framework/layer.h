#ifndef LAYER_H
#define LAYER_H
#include "base.h"
#include "tensor.h"
#include "perceptor.h"
#include <cudnn.h>


class Layer {
private:

protected:
	// Perceptor for this layer
	Perceptor* mPerceptor;

public:
	virtual ~Layer() {

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