#ifndef OP_LAYER_H
#define OP_LAYER_H

#include "layer.h"

class SoftmaxLayer : Layer {
private:

public:

	SoftmaxLayer() {

	}

	~SoftmaxLayer() {

	}

	void forwardPropagation();

	void backwardPropagation();
};

#endif