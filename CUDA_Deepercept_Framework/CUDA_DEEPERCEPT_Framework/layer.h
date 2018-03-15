#ifndef LAYER_H
#define LAYER_H
#include "base.h"


class Layer {
private:


public:
	Layer() {

	}
	
	~Layer() {

	}

	virtual void forwardPropagation();

	virtual void backwardPropagation();

};


#endif