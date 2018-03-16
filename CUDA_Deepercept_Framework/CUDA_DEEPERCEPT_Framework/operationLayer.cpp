#include "operationLayer.h"

void OperationLayer::forwardPropagation(Tensor* tInput, Tensor* tOutput) {

}

void OperationLayer::backPropagation(Tensor* gInput, Tensor* gOutput) {

}

void OperationLayer::forwardPropagation(Tensor* tInputA, Tensor* tInputB, Tensor* tOutput) {
	CuDNN_ERROR(
		cudnnOpTensor(
			mPerceptor->cudnnHandle(),
			mOpTensorDesc,
			&mAlpha,
			tInputA->descriptor(),
			tInputA->devDataPtr(),
			&mAlpha,
			tInputB->descriptor(),
			tInputB->devDataPtr(),
			&mBeta,
			tInputA->descriptor(),
			tOutput->devDataPtr()
		)
	);
	mInputA = tInputA;
	mInputB = tInputB;
	mOutput = tOutput;
}

void OperationLayer::backwardPropagation(Tensor* gInput, Tensor* gOutputA, Tensor* gOutputB) {
	switch (mOperation) {
	case CUDNN_OP_TENSOR_ADD:
		break;
	case CUDNN_OP_TENSOR_MUL:
		break;
	case CUDNN_OP_TENSOR_MIN:
		break;
	case CUDNN_OP_TENSOR_MAX:
		break;
	}
}