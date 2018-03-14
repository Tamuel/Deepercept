#ifndef BASICS_H
#define BASICS_H

#include <iostream>
#include <time.h>
#include <string>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <stdarg.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#define DECIMAL 10
#define BINARY 2

using namespace std;

typedef float dtype;


// Convert floating number to string
string f_to_s(dtype floatingNumber, int length = 10, int precision = 3, int floor = -5);

#endif