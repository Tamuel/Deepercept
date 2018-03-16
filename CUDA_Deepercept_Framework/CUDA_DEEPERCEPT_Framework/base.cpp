#include "base.h"

string f_to_s(dtype floatingNumber, int length, int precision, int floor) {
	char* fChar = new char[length + 1];
	float fractionalPart = floatingNumber - (long)floatingNumber;

	int _length = length;
	int _precision = precision;
	string _expression = "lf";

	floatingNumber = floatingNumber / pow(DECIMAL, floor);
	floatingNumber = long long(floatingNumber);
	floatingNumber *= pow(DECIMAL, floor);

	if (abs(floatingNumber) >= pow(DECIMAL, length)) {
		_length = length - 3;
		_precision = length - 7;
		_expression = "e";
	}
	else if (abs(floatingNumber) >= pow(DECIMAL, length - 2)) {
		_precision = 0;
	}

	if (floor >= 0)
		_precision = 0;

	string format = "%" + to_string(_length) + "." + to_string(_precision) + _expression;
	snprintf(fChar, length + 1, format.c_str(), floatingNumber);
	
	string out(fChar);

	delete[] fChar;

	return out;
}

void CuBLAS_ERROR(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return;
	case CUBLAS_STATUS_NOT_INITIALIZED:
		cerr << "cuBLAS : Library was not initialized." << endl;
		break;
	case CUBLAS_STATUS_ALLOC_FAILED:
		cerr << "cuBLAS : Resource allocation failed." << endl;
		break;
	case CUBLAS_STATUS_INVALID_VALUE:
		cerr << "cuBLAS : An unsupported value or parameter was passed to the function." << endl;
		break;
	case CUBLAS_STATUS_ARCH_MISMATCH:
		cerr << "cuBLAS : The function requires a feature absent from the device architecture;" <<
			" usually casued by the lack of support for double precision." << endl;
		break;
	case CUBLAS_STATUS_MAPPING_ERROR:
		cerr << "cuBLAS : An access to GPU memory space failed." << endl;
		break;
	case CUBLAS_STATUS_EXECUTION_FAILED:
		cerr << "cuBLAS : The GPU program failed to execute." << endl;
		break;
	case CUBLAS_STATUS_INTERNAL_ERROR:
		cerr << "cuBLAS : An internal cuBLAS operation failed." << endl;
		break;
	case CUBLAS_STATUS_NOT_SUPPORTED:
		cerr << "cuBLAS : The functionnality requested is not supported." << endl;
		break;
	case CUBLAS_STATUS_LICENSE_ERROR:
		cerr << "cuBLAS : The functionnality requested requires some license and an error" <<
			" was detected when trying to check the current licensing." << endl;
		break;
	}
	cerr << " File : " << __FILE__ << ", Line : " << __LINE__ << endl;
	exit(EXIT_FAILURE);
}

void CuDNN_ERROR(cudnnStatus_t error) {
	switch (error) {
	case CUDNN_STATUS_SUCCESS:
		return;
	case CUDNN_STATUS_NOT_INITIALIZED:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : Library was not initialized." << endl;
		break;
	case CUDNN_STATUS_INVALID_VALUE:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An incorrect value or parameter was passed." << endl;
		break;
	case CUDNN_STATUS_ALLOC_FAILED:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : Resource allocation failed." << endl;
		break;
	case CUDNN_STATUS_BAD_PARAM:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An incorrect value or parameter was passed." << endl;
		break;
	case CUDNN_STATUS_ARCH_MISMATCH:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The function requires a feature absent from the device architecture;" <<
			" usually casued by the lack of support for double precision." << endl;
		break;
	case CUDNN_STATUS_MAPPING_ERROR:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An access to GPU memory space failed." << endl;
		break;
	case CUDNN_STATUS_EXECUTION_FAILED:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The GPU program failed to execute." << endl;
		break;
	case CUDNN_STATUS_INTERNAL_ERROR:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : An internal cuBLAS operation failed." << endl;
		break;
	case CUDNN_STATUS_NOT_SUPPORTED:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The functionnality requested is not supported." << endl;
		break;
	case CUDNN_STATUS_LICENSE_ERROR:
		cerr << "cuDNN(" << cudnnGetErrorString(error) << ") : The functionnality requested requires some license and an error" <<
			" was detected when trying to check the current licensing." << endl;
		break;
	}
	cerr << " File : " << __FILE__ << ", Line : " << __LINE__ << endl;
	exit(EXIT_FAILURE);
}
