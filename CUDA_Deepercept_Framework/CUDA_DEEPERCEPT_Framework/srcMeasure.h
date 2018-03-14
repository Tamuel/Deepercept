#ifndef SRC_MEASURE_H
#define SRC_MEASURE_H

#include "base.h"
#include <chrono>
#include <cmath>

#define NUMBER_OF_TIMER 10

using namespace std;

class SrcMeasure {
private:

	// Chrono supported from C++ 11
	chrono::time_point<chrono::steady_clock> start_time[NUMBER_OF_TIMER];
	chrono::time_point<chrono::steady_clock> end_time[NUMBER_OF_TIMER];

public:
	SrcMeasure() {

	}

	// Start the timerIndex th timer
	void startTime(int timerIndex) {
		start_time[timerIndex] = chrono::high_resolution_clock::now();
	}

	// Return elapsed time of timerIndex th timer
	double endTime(int timerIndex, string s) {
		end_time[timerIndex] = chrono::high_resolution_clock::now();
		long long nsTimeCount = chrono::duration_cast<chrono::nanoseconds>(end_time[timerIndex] - start_time[timerIndex]).count();
		double elapsedTime = nsTimeCount / pow(10, 9);
		cout << "[" << s << "] Time : " << f_to_s(elapsedTime, 8, 6, -6) << endl;
		return elapsedTime;
	}
};

#endif