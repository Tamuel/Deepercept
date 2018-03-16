#ifndef OP_LAYER_H
#define OP_LAYER_H

#include "layer.h"

class OperationLayer : Layer {
private:
	// Descriptor for operation layer
	cudnnOpTensorDescriptor_t mOpTensorDesc;

	// Operation of this layer
	cudnnOpTensorOp_t mOperation;

	// Operation of this layer
	cudnnDataType_t mDatatype;

	// Operation of this layer
	cudnnNanPropagation_t mPropagateNan;

	// Tensor blender
	dtype mAlpha;

	// Tensor blender
	dtype mBeta;

	Tensor* mInputA;
	Tensor* mInputB;
	Tensor* mOutput;

public:
	OperationLayer(Perceptor* aPerceptor, cudnnOpTensorOp_t aOperation, cudnnDataType_t aDatatype = CUDNN_DATA_FLOAT,
		cudnnNanPropagation_t aPropagateNan = CUDNN_NOT_PROPAGATE_NAN) {

		mAlpha = 1; mBeta = 0;

		mPerceptor = aPerceptor;

		CuDNN_ERROR(cudnnCreateOpTensorDescriptor(&mOpTensorDesc));
		CuDNN_ERROR(
			cudnnSetOpTensorDescriptor(
				mOpTensorDesc,
				aOperation,
				aDatatype,
				aPropagateNan
			)
		);
		
		mOperation = aOperation;
		mDatatype = aDatatype;
		mPropagateNan = aPropagateNan;
	}

	~OperationLayer() {
		CuDNN_ERROR(cudnnDestroyOpTensorDescriptor(mOpTensorDesc));
	}

	virtual void forwardPropagation(Tensor* tInput, Tensor* tOutput);

	virtual void backPropagation(Tensor* gInput, Tensor* gOutput);

	virtual void forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput);

	virtual void backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB);

};

#endif