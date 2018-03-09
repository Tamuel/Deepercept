#include "basics.h"

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