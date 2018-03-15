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

void Tensor::printTensor(int floatLength, int floatPrecision, int floor) {
	cout << "  " << mName << " [";
	printShape();
	cout << "]" << endl;
	
	int maximumWidth = floatLength + 2;
	int indexWidth = floatLength;

	string gap = " ";

	// Actual output number of column and row
	int P_COL, P_ROW;

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
		if (col() > 10) P_COL = 10;
		else P_COL = col();

		if (row() > 10) P_ROW = 10;
		else P_ROW = row();

		// Print blank
		cout << setw(indexWidth) << "";

		// Print column index
		for (int col = 0; col < P_COL; col++) {
			cout << setw(indexWidth) << right << ("[" + to_string(col) + "]") << gap;
		}
		cout << endl;

		// Print data
		for (int row = 0; row < P_ROW; row++) {
			cout << setw(indexWidth) << right << ("[" + to_string(row) + "]"); // Print row index
			for (int col = 0; col < P_COL; col++)
				cout << setw(floatLength) << right << f_to_s((*this)(col, row), floatLength, floatPrecision, floor) << gap;
			cout << endl;
		}
		break;
	case 3:
		if (col() > 10) P_COL = 10;
		else P_COL = col();

		if (row() > 10) P_ROW = 10;
		else P_ROW = row();

		for (int z = 0; z < shape(0); z++) {
			// Print blank
			cout << setw(indexWidth) << ("[" + to_string(z) + "]");

			// Print column index
			for (int col = 0; col < P_COL; col++) {
				cout << setw(indexWidth) << right << ("[" + to_string(col) + "]") << gap;
			}
			cout << endl;

			// Print data
			for (int row = 0; row < P_ROW; row++) {
				cout << setw(indexWidth) << right << ("[" + to_string(row) + "]"); // Print row index
				for (int col = 0; col < P_COL; col++)
					cout << setw(floatLength) << right << f_to_s((*this)(z, col, row), floatLength, floatPrecision, floor) << gap;
				cout << endl;
			}
			
			cout << endl;
		}
		break;
	case 4: // NCHW
		if (col() > 10) P_COL = 10;
		else P_COL = col();

		if (row() > 10) P_ROW = 10;
		else P_ROW = row();

		for (int w = 0; w < shape(0); w++) {
			for (int z = 0; z < shape(1); z++) {
				// Print blank
				cout << setw(indexWidth) << ("[" + to_string(w) + ", " + to_string(z) + "]");

				// Print column index
				for (int col = 0; col < P_COL; col++) {
					cout << setw(indexWidth) << right << ("[" + to_string(col) + "]") << gap;
				}
				cout << endl;

				// Print data
				for (int row = 0; row < P_ROW; row++) {
					cout << setw(indexWidth) << right << ("[" + to_string(row) + "]"); // Print row index
					for (int col = 0; col < P_COL; col++)
						cout << setw(floatLength) << right << f_to_s((*this)(w, z, col, row), floatLength, floatPrecision, floor) << gap;
					cout << endl;
				}

				cout << endl;
			}
		}
		break;
	}
}

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