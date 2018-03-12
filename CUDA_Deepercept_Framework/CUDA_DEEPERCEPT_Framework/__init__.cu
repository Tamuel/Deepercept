#include "perceptor.h"
#include "srcMeasure.h"

int main(void) {
	Perceptor p1(1);
	Perceptor p2(0);
	SrcMeasure sm;
	p1.getGpuInformation();
	p1.getCuBlasVersion();
	p1.getCuDnnVersion();
	p1.getGpuDriverVersion();
	p1.setSynchronizeGpuStream(false);
	p2.setSynchronizeGpuStream(false);
	
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

	Tensor* m3 = p1.matAdd(&m1, &m2);
	m3->setName("m1 + m2");
	p1.retrievDataFromDevice(m3);
	m3->show();

	Tensor* m4 = p1.matMult(&m1, &m2);
	m4->setName("m1 x m2");
	p1.retrievDataFromDevice(m4);
	m4->show();

	Tensor* m5 = p1.matSub(&m1, &m2);
	m5->setName("m1 - m2");
	p1.retrievDataFromDevice(m5);
	m5->show();

	p1.matMult(2.0, &m1);
	p1.retrievDataFromDevice(&m1);
	m1.show();

	p1.matSwap(&m1, &m2);
	p1.retrievDataFromDevice(&m1);
	p1.retrievDataFromDevice(&m2);
	m1.show();
	m2.show();

	int i = p1.matMaxIndex(&m1);
	int j = p1.matMaxIndex(&m2);
	cout << "m1 max : " << i << endl;
	cout << "m2 max : " << j << endl;

	i = p1.matMinIndex(&m1);
	j = p1.matMinIndex(&m2);
	cout << "m1 min : " << i << endl;
	cout << "m2 min : " << j << endl;

	dtype sum = p1.matSum(&m1);
	dtype sum2 = p1.matSum(&m2);
	cout << "m1 sum : " << sum << endl;
	cout << "m2 sum : " << sum2 << endl;

	Tensor m6({ 1, 3 }, 1.0);
	Tensor m7({ 3, 1 }, 2.0);

	m6.show();
	m7.show();

	Tensor* m8 = p1.matMult(&m6, &m7);
	p1.retrievDataFromDevice(m8);
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
		p1.matTranspose(&m9);
		p1.retrievDataFromDevice(&m9);
		sm.endTime(0, "Transpose time");
		m9.show(9, 0);
	}
	
	Tensor m10({ 4096, 4096 });
	Tensor m11({ 4096, 4096 });
	for (int i = 0; i < 4096; i++)
		for (int j = 0; j < 4096; j++) {
			m10(i, j) = float(i) / float(j + 1);
			m11(j, i) = float(j) / float(i + 1);
		}
	Tensor m13({ 4096, 4096 });
	Tensor m14({ 4096, 4096 });
	for (int i = 0; i < 4096; i++)
		for (int j = 0; j < 4096; j++) {
			m13(i, j) = float(i) / float(j + 1);
			m14(j, i) = float(j) / float(i + 1);
		}

	p1.sendToDevice(&m10);
	p1.sendToDevice(&m11);
	p2.sendToDevice(&m13);
	p2.sendToDevice(&m14);

	Tensor* temp = new Tensor({ 4096, 4096 });
	p1.sendToDevice(temp);
	Tensor* temp2 = new Tensor({ 4096, 4096 });
	p2.sendToDevice(temp2);

	sm.startTime(0);
	for (int i = 0; i < 200; i++) {
		p1.matMult(temp, &m10, &m11);
		p2.matMult(temp2, &m13, &m14);
	}
	p1.retrievDataFromDevice(temp);
	p2.retrievDataFromDevice(temp2);
	double elapsedTime = sm.endTime(0, "Alloc time 2");

	temp->show();
	temp2->show();
}