// AutoGradient.cpp: 定义应用程序的入口点。
//

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "AutoGradient.h"

using namespace std;
using namespace autograd;
using namespace std::chrono;

int readInt(FILE *f) {
	const auto a = fgetc(f), b = fgetc(f), c = fgetc(f), d = fgetc(f);
	return (a << 24) | (b << 16) | (c << 8) | d;
}

Cube readMNISTImages(const char *filename) {
	const auto file = fopen(filename, "rb");
	const auto magic = readInt(file);
	assert(magic == 2051);
	const auto cnt = readInt(file);
	const auto rows = readInt(file);
	const auto cols = readInt(file);
	const auto size = static_cast<size_t>(cnt)* rows * cols;
	const auto buf = new unsigned char[size];
	const auto read = fread(buf, 1, size, file);
	assert(read == size);
	fclose(file);
	Cube ret;
	ret.reserve(cnt);
	auto ptr = buf;
	for (auto i = 0; i < cnt; i++) {
		Vector tmp(rows * cols);
		for (auto x = 0; x < rows * cols; x++)
			tmp(x) = static_cast<double>(*ptr++) / 255.0;
		ret.push_back(tmp);
	}
	delete[] buf;
	return ret;
}

Cube readMNISTLabels(const char *filename) {
	const auto file = fopen(filename, "rb");
	const auto magic = readInt(file);
	assert(magic == 2049);
	const auto cnt = readInt(file);
	const auto buf = new unsigned char[cnt];
	const auto read = fread(buf, 1, cnt, file);
	assert(read == cnt);
	fclose(file);
	Cube ret;
	ret.reserve(cnt);
	auto ptr = buf;
	for (auto i = 0; i < cnt; i++) {
		Vector tmp = Vector::Zero(10);
		tmp(*ptr++) = 1;
		ret.push_back(tmp);
	}
	delete[] buf;
	return ret;
}

// m is used to control the variance of random dist.
OpPtr dense(const OpPtr &prev, size_t prevSize, size_t thisSize, double m = 1) {
	auto v = sqrt(m / prevSize);
	auto w = parameter(randNormal(thisSize, prevSize, v));
	auto b = parameter(randNormal(thisSize, v));
	return w * prev + b;
}

bool correct(const Matrix &a, const Matrix &b) {
	Vector::Index ia, ib;
	static_cast<Vector>(a).maxCoeff(&ia);
	static_cast<Vector>(b).maxCoeff(&ib);
	return ia == ib;
}

int main() {
	const auto HIDDEN_SIZE = 128;
	const auto BATCH_SIZE = 32;

	auto x = constant(Vector::Zero(28 * 28));
	auto y = constant(Vector::Zero(10));
	auto h = dropout(mish(dense(x, 28 * 28, HIDDEN_SIZE, 2)), 0.2); // m = 2: Kaiming init.
	auto yHat = softmax(dense(h, HIDDEN_SIZE, 10));
	// Directly define cross-entropy loss function, no need for special op 
	auto loss = -dot(y, autograd::log(yHat)) - dot(1 - y, autograd::log(1 - yHat));
	// auto loss = crossEntropy(yHat, y);

	auto optimizer = make_shared<AdamOptimizer>(loss);
	cout << optimizer->graph() << endl;
	auto imagesTrain = readMNISTImages("D:/MNIST/train-images.idx3-ubyte");
	auto labelsTrain = readMNISTLabels("D:/MNIST/train-labels.idx1-ubyte");
	auto imagesTest = readMNISTImages("D:/MNIST/t10k-images.idx3-ubyte");
	auto labelsTest = readMNISTLabels("D:/MNIST/t10k-labels.idx1-ubyte");
	const auto sizeTrain = imagesTrain.size(), sizeTest = imagesTest.size();

	for (auto epoch = 1; epoch <= 100; epoch++) {
		double sumLoss = 0, accTrain = 0, accTest = 0;
		const auto start = high_resolution_clock::now();
		dynamic_pointer_cast<DropoutOp>(h)->setTraining(true);
		for (size_t i = 0; i < sizeTrain; i += BATCH_SIZE) {
			for (size_t j = i; j < sizeTrain && j < i + BATCH_SIZE; j++) {
				dynamic_pointer_cast<MatrixConstOp>(x)->set(imagesTrain[j]);
				dynamic_pointer_cast<MatrixConstOp>(y)->set(labelsTrain[j]);
				sumLoss += get<Scalar>(optimizer->propagate());
				accTrain += correct(get<Matrix>(optimizer->valueOf(yHat)), labelsTrain[j]);
			}
			optimizer->update();
			optimizer->clearGradient();
			printf("Epoch %3d: training %5.2lf%%\r", epoch, 100 * i / static_cast<double>(sizeTrain));
		}
		const double time = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
		dynamic_pointer_cast<DropoutOp>(h)->setTraining(false);
		for (size_t i = 0; i < sizeTest; i++) {
			dynamic_pointer_cast<MatrixConstOp>(x)->set(imagesTest[i]);
			dynamic_pointer_cast<MatrixConstOp>(y)->set(labelsTest[i]);
			optimizer->propagate(false);
			accTest += correct(get<Matrix>(optimizer->valueOf(yHat)), labelsTest[i]);
			printf("Epoch %3d: testing %5.2lf%% \r", epoch, 100 * i / static_cast<double>(sizeTest));
		}
		printf("Epoch %3d: avg loss %6.3lf train accuracy %5.2lf%% test accuracy %5.2lf%% tps %lfus\n",
			epoch, sumLoss / sizeTrain, 100 * accTrain / sizeTrain, 100 * accTest / sizeTest, time / sizeTrain);
	}
	return 0;
}
