#ifndef TENSOR_H
#define TENSOR_H
#include "base.h"

#if defined(NDEBUG) // If release mode
#define CUDA_CHECK(x) (x)
#else // If debug mode
#define CUDA_CHECK(x) do{\
	(x);\
	cudaError_t e = cudaGetLastError();\
	if(e != cudaSuccess) {\
		printf("cuda failure '%s' at %s:%d:\n",\
			cudaGetErrorString(e),\
			__FILE__, __LINE__);\
		exit(1);\
	}\
} while (0)
#endif


#define NAME_LENGTH 30
#define BLOCK_DIM 32
#define VECTOR_SIZE 8


enum tensorIndex {COL, ROW};

enum tensorDataType {FLOAT32, FLOAT64, INT32, INT64, BOOL};

enum tensorType {TENSOR, FILTER, CONV, ACTIVATION, POOLING, OP, REDUCE_TENSOR, CTC_LOSS, RNN, DROPOUT};

// Tensor class which is first class for data flow of deep learning framework
// With column major
class Tensor
{
private:
	// Tensor is friend of Perceptor class
	friend class Perceptor;

	// Number of tensors which generated
	static int num;

	// Name of tensor
	char mName[NAME_LENGTH]; // String object cannot run on CUDA device

	// Data type
	tensorDataType mDataType;

	// Shape of tensor
	// Ex) {3, 3, 3}
	int* mShape;

	// Dimension of tensor
	// Ex) 3
	int mDimension;

	// Whole size for tensor
	// Ex) 27
	int mSize;

	// For access data like tensor
	int* cumulatedDimension;

	// Is this just container for Device pointer
	bool isContainer;

	// Is this tensor have device pointer
	bool mHaveDevPtr;

	// Is this tensor have device pointer for data
	bool mHaveDevDataPtr;

	// Device ID for where this tensor be allocated
	int mDeviceId;

	// Pointer of this tensor in device
	Tensor* dev;

	// Pointer of this tensor data in device
	dtype* devData;

	// Pointer of this tensor in device
	int* devShape;

	// Pointer of this tensor data in device
	int* devCumulatedDimension;

	// Tensor descriptor for cuDNN
	cudnnTensorDescriptor_t tDesc;

	// Return real position of data with one dimenstional array
	template<typename... Args>
	__host__ __device__ int tensorPosition(int accessDimension, int pos, Args... args) {
		return pos * cumulatedDimension[accessDimension] + tensorPosition(accessDimension + 1, args...);
	}

	// Just return position
	__host__ __device__ int tensorPosition(int accessDimension, int pos) {
		return pos * cumulatedDimension[accessDimension];
	}

	void allocateData(dtype aInitValue = 0.0, bool toInitValue = true) {
		CUDA_CHECK(cudaMallocHost((void**)&data, sizeof(dtype) * mSize));
		if (toInitValue) {
			if (aInitValue == 0.0)
				memset(data, 0, sizeof(dtype) * mSize);
			else
				this->operator[](aInitValue);
		}
		isContainer = false;
	}

	void allocateData(dtype* aData) {
		CUDA_CHECK(cudaMallocHost((void**)&data, sizeof(dtype) * mSize));
		memcpy(data, aData, sizeof(dtype) * mSize);
		isContainer = false;
	}

	void initAttributes() {
		num++;
		mDataType = FLOAT32;
		dev = NULL;
		devData = NULL;
		mHaveDevPtr = false;
		mHaveDevDataPtr = false;
	}

	void setCumulatedDimension();

	void setShape(const initializer_list<int>& aShape);

	void createTensorDescriptor() {
		cudnnCreateTensorDescriptor(&tDesc);
		cudnnSetTensor4dDescriptor(
			tDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n(), c(), h(), w()
		);
	}

public:
	// Tensor data with dtype
	dtype* data;

	Tensor() : isContainer(true) {}

	/*
	Tensor really consist of one dimensional data,
	But it act like tensor with shape.
	Tensor data is initialized to 0.
	If dimension of tensor is 4 then it generate tensor descriptor for cuDNN with NCHW format
	*/
	Tensor(const initializer_list<int>& aShape, string aName = "Tensor" + to_string(num),
		dtype aInitValue = 0.0, bool toInitValue = true) : mSize(1), mDeviceId(0) {
		initAttributes();
		setName(aName);
		setShape(aShape);
		setCumulatedDimension();
		allocateData(aInitValue, toInitValue);
		if(mDimension == 4) createTensorDescriptor();
	}

	Tensor(const initializer_list<int>& aShape, dtype aInitValue) :
		Tensor(aShape, "Tensor" + to_string(num), aInitValue, true) {}

	Tensor(const initializer_list<int>& aShape, double aInitValue) :
		Tensor(aShape, "Tensor" + to_string(num), dtype(aInitValue), true) {}

	Tensor(const initializer_list<int>& aShape, bool toInitValue) :
		Tensor(aShape, "Tensor" + to_string(num), 0.0, toInitValue) {}

	// Copy Constructor
	Tensor(const Tensor& other) {
		initAttributes();
		string temp = "Tensor" + to_string(num);
		setName(temp);
		mDimension = other.mDimension;
		mSize = other.mSize;
		mShape = new int[mDimension];
		memcpy(mShape, other.mShape, sizeof(int) * mDimension);
		cumulatedDimension = new int[mDimension];
		memcpy(cumulatedDimension, other.cumulatedDimension, sizeof(int) * mDimension);

		allocateData(other.data);
	}

	~Tensor() {
		if (!isContainer) {
			freeDevAndDevData();
			cudaFreeHost(data);
			delete[] mShape;
			delete[] cumulatedDimension;
		}
	}

	// Return is shape is same or not
	bool isSame(Tensor& other);


	// Set data by pData array
	void operator[](dtype* aData) {
		memcpy(data, aData, sizeof(dtype) * size());
	}

	// Set data by pValue value
	void operator[](dtype aValue);

	void operator[](const initializer_list<dtype>& aData);

	__host__ __device__ const int dimension() {
		return mDimension;
	}

	__host__ __device__ const int* shape() {
		return mShape;
	}

	__host__ __device__ const int shape(int i) {
		return mShape[i];
	}

	__host__ __device__ const int size() {
		return mSize;
	}

	const string name() {
		return string(mName);
	}

	const tensorDataType type() {
		return mDataType;
	}

	const int row() {
		if (mDimension - 2 < 0) {
			cout << "Cannot access row with " << mDimension << "-dimension tensor" << endl;
			exit(EXIT_FAILURE);
		}
		return shape(ROW + mDimension - 2);
	}

	const int col() {
		if (mDimension - 2 < 0) {
			cout << "Cannot access col with " << mDimension << "-dimension tensor" << endl;
			exit(EXIT_FAILURE);
		}
		return shape(COL + mDimension - 2);
	}

	// Get number of images when this tensor format is NCHW
	const int n() {
		if (mDimension < 4) {
			cout << "Cannot access n with " << mDimension << "-dimension tensor" << endl;
			exit(EXIT_FAILURE);
		}
		return shape(mDimension - 4);
	}

	// Get number of channel when this tensor format is NCHW
	const int c() {
		if (mDimension < 3) {
			cout << "Cannot access n with " << mDimension << "-dimension tensor" << endl;
			exit(EXIT_FAILURE);
		}
		return shape(mDimension - 3);
	}

	// Get height when this tensor format is NCHW
	const int h() {
		if (mDimension < 2) {
			cout << "Cannot access n with " << mDimension << "-dimension tensor" << endl;
			exit(EXIT_FAILURE);
		}
		return shape(mDimension - 2);
	}

	// Get width when this tensor format is NCHW
	const int w() {
		if (mDimension < 1) {
			cout << "Cannot access n with " << mDimension << "-dimension tensor" << endl;
			exit(EXIT_FAILURE);
		}
		return shape(mDimension - 1);
	}

	const bool haveDevicePtr() {
		return mHaveDevPtr;
	}

	const bool haveDeviceDataPtr() {
		return mHaveDevDataPtr;
	}

	void setName(string newName) {
		strncpy(mName, newName.c_str(), sizeof(char) * newName.size() + 1);
		mName[sizeof(newName) - 1] = 0;
	}

	template<typename... Args>
	__host__ __device__ dtype& operator()(int i, Args... args) {
		int pos = tensorPosition(0, i, args...);
		if (pos >= mSize) {
			printf("Cannot access %d element of %s!\n", pos, mName);
			//exit(1);
		}
		return data[pos];
	}

	__host__ __device__ dtype& operator()(int i) {
		if (i >= mSize) {
			printf("Cannot access %d element of %s!\n", i, mName);
			//exit(1);
		}
		return data[i];
	}

	// Get device pointer for this tensor
	Tensor* devPtr() {
		if (mHaveDevPtr)
			return dev;
		else {
			cout << "There are no device pointer for Tensor [" << mName << "]" << endl;
			exit(1);
		}
	}

	// Get device pointer for this tensor data
	dtype* devDataPtr() {
		if (mHaveDevDataPtr)
			return devData;
		else {
			cout << "There are no device pointer for Tensor data [" << mName << "]" << endl;
			exit(1);
		}
	}

	cudnnTensorDescriptor_t descriptor() {
		return tDesc;
	}

	// Deallocate device data
	void freeDevAndDevData() {
		mHaveDevPtr = false;
		mHaveDevDataPtr = false;
		cudaFree(devShape);
		cudaFree(devCumulatedDimension);
		cudaFree(devData);
		cudaFree(dev);
	}

	// Deallocate device data
	void freeDevData() {
		mHaveDevDataPtr = false;
		cudaFree(devData);
	}

	void setDevice(int aDeviceId) {
		mDeviceId = aDeviceId;
	}

	// Device ID for where this tensor will be allocated
	int deviceId() {
		return mDeviceId;
	}
	
	// Show tensor data, if you want to see current data in device you need to retreive data first (retrieveDataFromDevice)
	// Shape printed reversely for natural perceive (WHCN)
	void printTensor(int floatLength = 9, int floatPrecision = 3, int floor = -4);

	// Show tensor data, if you want to see current data in device you need to retreive data first (retrieveDataFromDevice)
	// Shape printed reversely for natural perceive (WHCN)
	void printData() {
		cout << name() << " ["; printShape(); cout << "]";
		for (int i = 0; i < mSize; i++) {
			cout << f_to_s(data[i]) << "  ";
			if ((i + 1) % 8 == 0)
				cout << endl;
		}
		cout << endl;
	}

	// Shape printed reversely for natural perceive (WHCN)
	void printShape() {
		for (int i = mDimension - 1; i >= 0; i--)
			if(i != 0)
				cout << shape(i) << " x ";
			else
				cout << shape(i);
	}

	// Swap shape[dim1] and shape[dim2]
	void swapDimension(int dim1, int dim2);

	// Reshape current tensor, if forceReshape is true than you can forcedly reshape tensor, but can loose data
	void reshape(const initializer_list<int>& aShape, bool forceReshape = false);

};

#endif