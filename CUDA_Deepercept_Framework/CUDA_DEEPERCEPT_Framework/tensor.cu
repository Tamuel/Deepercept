#include "tensor.h"
int Tensor::num = 0;

__global__ void setDataKernel(Tensor* dev, dtype pValue) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dev->size())
		(*dev)(i) = pValue;
}

void Tensor::operator[](dtype aValue) {
	if (mSize >= 10000) { // Run on GPU
		sendToDevice(false);
		setDataKernel <<<mSize / (BLOCK_DIM * BLOCK_DIM) + 1, BLOCK_DIM * BLOCK_DIM >>> (dev, aValue);
		cudaThreadSynchronize();
		retrievDataFromDevice(dev);
		freeDevAndDevData();
	}
	else { // Run on CPU
		for (int i = 0; i < mSize; i++)
			data[i] = aValue;
	}
}

void Tensor::operator[](const initializer_list<dtype>& aData) {
	if (aData.size() == size())
		copy(aData.begin(), aData.end(), data);
	else {
		printf("%s[size = %d] and input data[size = %d] size are different \n", mName, size(), aData.size());
		exit(1);
	}
}

void Tensor::show(int floatLength, int floatPrecision, int floor) {
	cout << "  " << mName << " [";
	for (int i = 0; i < mDimension; i++) {
		cout << mShape[i];
		if (i != mDimension - 1)
			cout << " x ";
	}
	cout << "]" << endl;
	
	int maximumWidth = floatLength + 2;
	int indexWidth = floatLength - 2;

	switch (mDimension) {
	case 1:
		if (mSize <= maximumWidth) {
			cout << setw(indexWidth) << "" << " ";
			for (int i = 0; i < mShape[0]; i++) {
				string temp = "[" + to_string(i) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << endl;

			cout << setw(indexWidth) << "[0]" << " ";
			for (int i = 0; i < mSize; i++)
				cout << setw(floatLength) << f_to_s((*this)(i), floatLength, floatPrecision, floor) << "  ";
			cout << endl;
		}
		else {
			cout << setw(indexWidth) << "" << " ";
			for (int i = 0; i < maximumWidth / 2; i++) {
				string temp = "[" + to_string(i) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << setw(floatLength) << right << "..." << "  ";
			for (int i = mShape[0] - maximumWidth / 2; i < mShape[0]; i++) {
				string temp = "[" + to_string(i) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << endl;

			cout << setw(indexWidth) << "[0]" << " ";
			for (int i = 0; i < maximumWidth / 2; i++)
				cout << setw(floatLength) << f_to_s((*this)(i), floatLength, floatPrecision, floor) << "  ";

			cout << setw(floatLength) << right << "..." << "  ";

			for (int i = mShape[0] - maximumWidth / 2; i < mShape[0]; i++)
				cout << setw(floatLength) << f_to_s((*this)(i), floatLength, floatPrecision, floor) << "  ";
			
			cout << endl;
		}
		break;

	case 2:
		if (mShape[0] <= maximumWidth && mShape[1] <= maximumWidth) {
			cout << setw(indexWidth) << "" << " ";
			for (int i = 0; i < mShape[1]; i++) {
				string temp = "[" + to_string(i) + "]";
				cout << setw(floatLength) << right << temp << "  ";
			}
			cout << endl;

			for (int i = 0; i < mShape[0]; i++) {
				string temp = "[" + to_string(i) + "]";
				cout << setw(indexWidth) << temp << " ";
				for (int j = 0; j < mShape[1]; j++)
					cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";
				cout << endl;
			}
		}
		else {
			if (mShape[0] > maximumWidth && mShape[1] > maximumWidth) {
				cout << setw(indexWidth) << "" << " ";
				for (int i = 0; i < maximumWidth / 2; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(floatLength) << right << temp << "  ";
				}
				cout << setw(floatLength) << right << "..." << "  ";
				for (int i = mShape[1] - maximumWidth / 2; i < mShape[1]; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(floatLength) << right << temp << "  ";
				}
				cout << endl;
				for (int i = 0; i < maximumWidth / 2; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(indexWidth) << temp << " ";

					for (int j = 0; j < maximumWidth / 2; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";

					cout << setw(floatLength) << right << "..." << "  ";

					for (int j = mShape[1] - maximumWidth / 2; j < mShape[1]; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";

					cout << endl;
				}

				for (int j = 0; j < 3; j++) {
					cout << setw(indexWidth) << "." << " ";
					for (int i = 0; i < maximumWidth; i++)
						cout << setw(floatLength) << right << "." << "  ";
					cout << endl;
				}

				for (int i = mShape[0] - maximumWidth / 2; i < mShape[0]; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(indexWidth) << temp << " ";

					for (int j = 0; j < maximumWidth / 2; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";

					cout << setw(floatLength) << right << "..." << "  ";

					for (int j = mShape[1] - maximumWidth / 2; j < mShape[1]; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";

					cout << endl;
				}
			}
			else if (mShape[0] <= maximumWidth && mShape[1] > maximumWidth) {
				cout << setw(indexWidth) << "" << " ";
				for (int i = 0; i < maximumWidth / 2; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(floatLength) << right << temp << "  ";
				}
				cout << setw(floatLength) << right << "..." << "  ";
				for (int i = mShape[1] - maximumWidth / 2; i < mShape[1]; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(floatLength) << right << temp << "  ";
				}
				cout << endl;

				for (int i = 0; i < mShape[0]; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(indexWidth) << temp << " ";

					for (int j = 0; j < maximumWidth / 2; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";

					cout << setw(floatLength) << right << "..." << "  ";

					for (int j = mShape[1] - maximumWidth / 2; j < mShape[1]; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";

					cout << endl;
				}
			}
			else if (mShape[0] > maximumWidth && mShape[1] <= maximumWidth) {
				cout << setw(indexWidth) << "" << " ";
				for (int i = 0; i < mShape[1]; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(floatLength) << right << temp << "  ";
				}
				cout << endl;

				for (int i = 0; i < maximumWidth / 2; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(indexWidth) << temp << " ";
					for (int j = 0; j < mShape[1]; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";
					cout << endl;
				}

				for (int j = 0; j < 3; j++) {
					cout << setw(indexWidth) << "." << " ";
					for (int i = 0; i < mShape[1]; i++)
						cout << setw(floatLength) << right << "." << "  ";
					cout << endl;
				}

				for (int i = mShape[0] - maximumWidth / 2; i < mShape[0]; i++) {
					string temp = "[" + to_string(i) + "]";
					cout << setw(indexWidth) << temp << " ";
					for (int j = 0; j < mShape[1]; j++)
						cout << setw(floatLength) << f_to_s((*this)(i, j), floatLength, floatPrecision, floor) << "  ";
					cout << endl;
				}
			}
		}
		break;
	}
}

void Tensor::sendToDevice(bool sendData) {
	Tensor* devPtr = 0;
	CUDA_CHECK(cudaMalloc((void**)&devPtr, sizeof(Tensor)));
	CUDA_CHECK(cudaMemcpy(devPtr, this, sizeof(Tensor), cudaMemcpyHostToDevice));

	int* hostShape;
	CUDA_CHECK(cudaMalloc((void**)&hostShape, sizeof(int) * this->dimension()));
	CUDA_CHECK(cudaMemcpy(hostShape, this->mShape, sizeof(int) * this->dimension(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&(devPtr->mShape), &hostShape, sizeof(int*), cudaMemcpyHostToDevice));

	int* hostCumulatedDimension;
	CUDA_CHECK(cudaMalloc((void**)&hostCumulatedDimension, sizeof(int) * this->dimension()));
	CUDA_CHECK(cudaMemcpy(hostCumulatedDimension, this->cumulatedDimension, sizeof(int) * this->dimension(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&(devPtr->cumulatedDimension), &hostCumulatedDimension, sizeof(int*), cudaMemcpyHostToDevice));
	
	// Set device tensor as container
	CUDA_CHECK(cudaMemcpy(&devPtr->isContainer, new bool(true), sizeof(bool), cudaMemcpyHostToDevice));

	// Copy host data to device
	dtype* hostData;
	CUDA_CHECK(cudaMalloc((void**)&hostData, sizeof(dtype) * this->size()));
	if (sendData)
		CUDA_CHECK(cudaMemcpy(hostData, this->data, sizeof(dtype) * this->size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&(devPtr->data), &hostData, sizeof(dtype*), cudaMemcpyHostToDevice));
	devData = hostData;
	dev = devPtr;
	devShape = hostShape;
	devCumulatedDimension = hostCumulatedDimension;
	mHaveDevPtr = true;
	mHaveDevDataPtr = true;
}

void Tensor::sendDataToDevice() {
	dtype* devPtr = 0;
	CUDA_CHECK(cudaMalloc((void**)&devPtr, sizeof(dtype) * size()));
	CUDA_CHECK(cudaMemcpy(devPtr, data, sizeof(dtype) * size(), cudaMemcpyHostToDevice));
	devData = devPtr;
	mHaveDevDataPtr = true;
}

void Tensor::retrievDataFromDevice(bool retreiveOnlyData) {
	CUDA_CHECK(cudaMemcpy(data, devData, this->size() * sizeof(dtype), cudaMemcpyDeviceToHost));
	if (!retreiveOnlyData) {
		Tensor temp;
		CUDA_CHECK(cudaMemcpy(&temp, dev, sizeof(Tensor), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(cumulatedDimension, temp.cumulatedDimension, this->dimension() * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(mShape, temp.mShape, this->dimension() * sizeof(int), cudaMemcpyDeviceToHost));
		temp.setName("");
	}
}

bool Tensor::isSame(Tensor& other) {
	for (int i = 0; i < mDimension; i++)
		if (mShape[i] != other.shape()[i])
			return false;

	return true;
}

void Tensor::swapDimension(int dim1, int dim2) {
	if (dim1 >= mDimension || dim1 < 0 || dim2 >= mDimension || dim2 < 0) {
		cout << "Cannot access " << mName << " " << dim1 << " or " << dim2 << endl;
		exit(EXIT_FAILURE);
	}

	int temp = mShape[dim1];
	mShape[dim1] = mShape[dim2];
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
	for (int i = 0; i < mDimension; i++) {
		int tempCumDim = 1;
		for (int j = i + 1; j < mDimension; j++)
			tempCumDim *= mShape[j];
		cumulatedDimension[i] = tempCumDim;
	}
}