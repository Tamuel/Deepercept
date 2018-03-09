#include "perceptor.h"
#include "srcMeasure.h"

int main(void) {
	
	Perceptor p;
	SrcMeasure sm;
	p.getGpuInformation();
	p.getCuBlasVersion();
	p.getCuDnnVersion();
	p.getGpuDriverVersion();
	p.setSynchronizeGpuStream(true);
	
	Tensor m1({ 3, 3 }, "m1", 1.0);
	Tensor m2({ 3, 3 }, "m2", 1.0);

	m1[{ 9, 2, 3,
		4, 5, 6,
		7, 8, 9}];

	m2[{ 1, 0, 0,
		0, 1, 0,
		0, 0, 1}];

	m1.show();
	m2.show();

	Tensor* m3 = p.matAdd(&m1, &m2);
	m3->setName("m1 + m2");
	m3->retrievDataFromDevice();
	m3->show();

	Tensor* m4 = p.matMult(&m1, &m2);
	m4->setName("m1 x m2");
	m4->retrievDataFromDevice();
	m4->show();

	Tensor* m5 = p.matSub(&m1, &m2);
	m5->setName("m1 - m2");
	m5->retrievDataFromDevice();
	m5->show();

	p.matMult(2.0, &m1);
	m1.retrievDataFromDevice();
	m1.show();

	p.matSwap(&m1, &m2);
	m1.retrievDataFromDevice();
	m2.retrievDataFromDevice();
	m1.show();
	m2.show();

	int i = p.matMaxIndex(&m1);
	int j = p.matMaxIndex(&m2);
	cout << "m1 max : " << i << endl;
	cout << "m2 max : " << j << endl;

	i = p.matMinIndex(&m1);
	j = p.matMinIndex(&m2);
	cout << "m1 min : " << i << endl;
	cout << "m2 min : " << j << endl;

	dtype sum = p.matSum(&m1);
	dtype sum2 = p.matSum(&m2);
	cout << "m1 sum : " << sum << endl;
	cout << "m2 sum : " << sum2 << endl;

	Tensor m6({ 1, 3 }, 1.0);
	Tensor m7({ 3, 1 }, 2.0);

	m6.show();
	m7.show();

	Tensor* m8 = p.matMult(&m6, &m7);
	m8->retrievDataFromDevice();
	m8->show();

	int size1 = 4096;
	int size2 = 4096;

	Tensor m12({ 4096, 4096 });
	Tensor m9({ size1, size2 }, 1.0);
	for (int i = 0; i < size1; i++) 
		for (int j = 0; j < size2; j++)
			m9(i, j) = i * size2 + j;

	m9.show(9, 0);
	for (int i = 0; i < 10; i++) {
		sm.startTime(0);
		p.matTranspose(&m9);
		m9.retrievDataFromDevice();
		sm.endTime(0, "Transpose time");
		m9.show(9, 0);
	}
	
	Tensor m10({ 4096, 4096 });
	Tensor m11({ 4096, 4096 });
	for (int i = 0; i < 4096; i++)
		for (int j = 0; j < 4096; j++) {
			m10(i, j) = i * 4096 + j;
			m11(j, i) = i * 4096 + j;
		}

	m10.show();
	m11.show();
	sm.startTime(0);
	p.matEltMult(&m12, &m10, &m11);
	m12.retrievDataFromDevice();
	sm.endTime(0, "Elt Mult time");
	sm.startTime(0);
	p.matEltMult(&m12, &m10, &m11);
	m12.retrievDataFromDevice();
	sm.endTime(0, "Elt Mult time");
	m12.show();
	dtype* data;
	sm.startTime(0);
	cudaMalloc((void**)&data, sizeof(dtype) * 4096 * 4096);
	double elapsedTime = sm.endTime(0, "Alloc time");
	cout << 1 / elapsedTime << endl;
}