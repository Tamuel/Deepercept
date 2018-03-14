#include "tensor.h"
int Tensor::num = 0;

__global__ void setDataKernel(Tensor* dev, dtype pValue) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dev->size())
		(*dev)(i) = pValue;
}

void Tensor::operator[](dtype aValue) {
	for (int i = 0; i < mSize; i++)
		data[i] = aValue;
}

void Tensor::operator[](const initializer_list<dtype>& aData) {
	if (aData.size() == size())
		copy(aData.begin(), aData.end(), data);
	else {
		printf("%s[size = %d] and input data[size = %d] size are different \n", mName, size(), aData.size());
		exit(1);
	}
}

void Tensor::print(bool NHWC, int floatLength, int floatPrecision, int floor) {
	cout << "  " << mName << " [";
	printShape();
	cout << "]" << endl;
	
	int maximumWidth = floatLength + 2;
	int indexWidth = floatLength;
	switch (mDimension) {
	case 1:
		if (mSize <= maximumWidth) {
			cout << setw(indexWidth) << "" << " ";
			for (int x = 0; x < shape(0); x++) {
				string temp = "[" + to_string(x) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << endl;

			cout << setw(indexWidth) << "[0]" << " ";
			for (int x = 0; x < mSize; x++)
				cout << setw(floatLength) << f_to_s((*this)(x), floatLength, floatPrecision, floor) << "  ";
			cout << endl;
		}
		else {
			cout << setw(indexWidth) << "" << " ";
			for (int x = 0; x < maximumWidth / 2; x++) {
				string temp = "[" + to_string(x) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << setw(floatLength) << right << "..." << "  ";
			for (int x = shape(0) - maximumWidth / 2; x < shape(0); x++) {
				string temp = "[" + to_string(x) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << endl;

			cout << setw(indexWidth) << "[0]" << " ";
			for (int x = 0; x < maximumWidth / 2; x++)
				cout << setw(floatLength) << f_to_s((*this)(x), floatLength, floatPrecision, floor) << "  ";

			cout << setw(floatLength) << right << "..." << "  ";

			for (int x = shape(0) - maximumWidth / 2; x < shape(0); x++)
				cout << setw(floatLength) << f_to_s((*this)(x), floatLength, floatPrecision, floor) << "  ";

			cout << endl;
		}
		break;

	case 2:
		// Print blank
		cout << setw(indexWidth) << "";

		// Print column index
		for (int col = 0; col < shape(COL); col++) {
			cout << setw(indexWidth) << right << ("[" + to_string(col) + "]");
		}
		cout << endl;

		// Print data
		for (int row = 0; row < shape(ROW); row++) {
			cout << setw(indexWidth) << right << ("[" + to_string(row) + "]"); // Print row index
			for (int col = 0; col < shape(COL); col++)
				cout << setw(floatLength) << right << f_to_s((*this)(col, row), floatLength, floatPrecision, floor);
			cout << endl;
		}
		break;
	case 3:
		for (int z = 0; z < shape(0); z++) {
			// Print blank
			cout << setw(indexWidth) << ("[" + to_string(z) + "]");

			// Print column index
			for (int col = 0; col < shape(COL + 1); col++) {
				cout << setw(indexWidth) << right << ("[" + to_string(col) + "]");
			}
			cout << endl;

			// Print data
			for (int row = 0; row < shape(ROW + 1); row++) {
				cout << setw(indexWidth) << right << ("[" + to_string(row) + "]"); // Print row index
				for (int col = 0; col < shape(COL + 1); col++)
					cout << setw(floatLength) << right << f_to_s((*this)(z, col, row), floatLength, floatPrecision, floor);
				cout << endl;
			}
			
			cout << endl;
		}
		break;
	case 4:
		for (int w = 0; w < shape(0); w++) {
			for (int z = 0; z < shape(1); z++) {
				// Print blank
				cout << setw(indexWidth) << ("[" + to_string(w) + ", " + to_string(z) + "]");

				// Print column index
				for (int col = 0; col < shape(COL + 2); col++) {
					cout << setw(indexWidth) << right << ("[" + to_string(col) + "]");
				}
				cout << endl;

				// Print data
				for (int row = 0; row < shape(ROW + 2); row++) {
					cout << setw(indexWidth) << right << ("[" + to_string(row) + "]"); // Print row index
					for (int col = 0; col < shape(COL + 2); col++)
						cout << setw(floatLength) << right << f_to_s((*this)(w, z, col, row), floatLength, floatPrecision, floor);
					cout << endl;
				}

				cout << endl;
			}
		}
		break;
	}
}

//void Tensor::sendToDevice(bool sendData) {
//	Tensor* devPtr = 0;
//	cudaSetDevice(mDeviceId);
//	CUDA_CHECK(cudaMalloc((void**)&devPtr, sizeof(Tensor)));
//	CUDA_CHECK(cudaMemcpy(devPtr, this, sizeof(Tensor), cudaMemcpyHostToDevice));
//
//	int* hostShape;
//	CUDA_CHECK(cudaMalloc((void**)&hostShape, sizeof(int) * this->dimension()));
//	CUDA_CHECK(cudaMemcpy(hostShape, this->mShape, sizeof(int) * this->dimension(), cudaMemcpyHostToDevice));
//	CUDA_CHECK(cudaMemcpy(&(devPtr->mShape), &hostShape, sizeof(int*), cudaMemcpyHostToDevice));
//
//	int* hostCumulatedDimension;
//	CUDA_CHECK(cudaMalloc((void**)&hostCumulatedDimension, sizeof(int) * this->dimension()));
//	CUDA_CHECK(cudaMemcpy(hostCumulatedDimension, this->cumulatedDimension, sizeof(int) * this->dimension(), cudaMemcpyHostToDevice));
//	CUDA_CHECK(cudaMemcpy(&(devPtr->cumulatedDimension), &hostCumulatedDimension, sizeof(int*), cudaMemcpyHostToDevice));
//	
//	// Set device tensor as container
//	CUDA_CHECK(cudaMemcpy(&devPtr->isContainer, new bool(true), sizeof(bool), cudaMemcpyHostToDevice));
//
//	// Copy host data to device
//	dtype* hostData;
//	CUDA_CHECK(cudaMalloc((void**)&hostData, sizeof(dtype) * this->size()));
//	if (sendData)
//		CUDA_CHECK(cudaMemcpy(hostData, this->data, sizeof(dtype) * this->size(), cudaMemcpyHostToDevice));
//	CUDA_CHECK(cudaMemcpy(&(devPtr->data), &hostData, sizeof(dtype*), cudaMemcpyHostToDevice));
//	devData = hostData;
//	dev = devPtr;
//	devShape = hostShape;
//	devCumulatedDimension = hostCumulatedDimension;
//	mHaveDevPtr = true;
//	mHaveDevDataPtr = true;
//}
//
//void Tensor::sendDataToDevice() {
//	if (haveDevicePtr()) {
//		dtype* devPtr = 0;
//		cudaSetDevice(mDeviceId);
//		cudaFree(devData);
//		CUDA_CHECK(cudaMalloc((void**)&devPtr, sizeof(dtype) * size()));
//		CUDA_CHECK(cudaMemcpy(devPtr, data, sizeof(dtype) * size(), cudaMemcpyHostToDevice));
//		devData = devPtr;
//		mHaveDevDataPtr = true;
//	}
//	else {
//		cout << name() << " is not allocated at device" << endl;
//		exit(EXIT_FAILURE);
//	}
//}
//
//void Tensor::retrievDataFromDevice(bool retreiveOnlyData) {
//	if (haveDevicePtr() && haveDeviceDataPtr()) {
//		cudaSetDevice(mDeviceId);
//		CUDA_CHECK(cudaMemcpy(data, devData, this->size() * sizeof(dtype), cudaMemcpyDeviceToHost));
//		if (!retreiveOnlyData) {
//			Tensor temp;
//			CUDA_CHECK(cudaMemcpy(&temp, dev, sizeof(Tensor), cudaMemcpyDeviceToHost));
//			CUDA_CHECK(cudaMemcpy(cumulatedDimension, temp.cumulatedDimension, this->dimension() * sizeof(int), cudaMemcpyDeviceToHost));
//			CUDA_CHECK(cudaMemcpy(mShape, temp.mShape, this->dimension() * sizeof(int), cudaMemcpyDeviceToHost));
//			temp.setName("");
//		}
//	}
//}

bool Tensor::isSame(Tensor& other) {
	for (int i = 0; i < mDimension; i++)
		if (shape(i) != other.shape(i))
			return false;

	return true;
}

void Tensor::swapDimension(int dim1, int dim2) {
	cudaSetDevice(mDeviceId);
	if (dim1 >= mDimension || dim1 < 0 || dim2 >= mDimension || dim2 < 0) {
		cout << "Cannot access " << mName << " " << dim1 << " or " << dim2 << endl;
		exit(EXIT_FAILURE);
	}

	int temp = shape(dim1);
	mShape[dim1] = shape(dim2);
	mShape[dim2] = temp;

	setCumulatedDimension();

	CUDA_CHECK(cudaMemcpy(devShape, mShape, sizeof(int) * dimension(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(devCumulatedDimension, cumulatedDimension, sizeof(int) * dimension(), cudaMemcpyHostToDevice));
}

void Tensor::setShape(const initializer_list<int>& aShape) {
	mDimension = aShape.size();
	mShape = new int[mDimension];
	copy(aShape.begin(), aShape.end(), mShape);
	mSize = 1;
	initializer_list<int>::iterator iter = aShape.begin();
	while (iter != aShape.end()) {
		mSize *= *iter;
		iter++;
	}
}

void Tensor::reshape(const initializer_list<int>& aShape, bool forceReshape) {
	int tempSize = 1;
	initializer_list<int>::iterator iter = aShape.begin();
	while (iter != aShape.end()) {
		tempSize *= *iter;
		iter++;
	}

	if (!forceReshape && tempSize != mSize) {
		cout << "Size of new shape is different from original tensor size " << tempSize << " != " << mSize <<
			", if you want to change size forcedly you need to set argument [forceReshape = true]." << endl;
		exit(1);
	}
	delete[] mShape;
	delete[] cumulatedDimension;
	mDimension = aShape.size();
	mShape = new int[mDimension];
	copy(aShape.begin(), aShape.end(), mShape);
	mSize = tempSize;
	setCumulatedDimension();
}



void Tensor::setCumulatedDimension() {
	cumulatedDimension = new int[mDimension];
	cumulatedDimension[mDimension - 1] = 1;
	for (int i = mDimension - 2; i >= 0; i--) {
		cumulatedDimension[i] = cumulatedDimension[i + 1] * shape(i + 1);
	}
}