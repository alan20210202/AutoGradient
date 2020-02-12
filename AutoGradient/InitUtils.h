#ifndef AUTOGRADIENT_INITUTILS_H
#define AUTOGRADIENT_INITUTILS_H
	
#include "Value.h"
#include "Random.h"

namespace autograd {
	inline Matrix randNormal(const size_t rows, const size_t cols, const double var = 1, const double mean = 0) {
		std::mt19937_64 gen = seededRNG();
		std::normal_distribution<Scalar> dist(mean, var);
		Matrix ret(rows, cols);
		for (size_t i = 0; i < rows; i++)
			for (size_t j = 0; j < cols; j++)
				ret(i, j) = dist(gen);
		return ret;
	}

	inline Vector randNormal(const size_t cols, const double var = 1, const double mean = 0) {
		std::mt19937_64 gen = seededRNG();
		std::normal_distribution<Scalar> dist(mean, var);
		Vector ret(cols);
		for (size_t i = 0; i < cols; i++)
			ret(i)  = dist(gen);
		return ret;
	}

	inline Matrix randUniform(const size_t rows, const size_t cols, const double margin = 1, const double mean = 0) {
		std::mt19937_64 gen = seededRNG();
		const std::uniform_real_distribution<Scalar> dist(mean - margin, mean + margin);
		Matrix ret(rows, cols);
		for (size_t i = 0; i < rows; i++)
			for (size_t j = 0; j < cols; j++)
				ret(i, j) = dist(gen);
		return ret;
	}

	inline Vector randUniform(const size_t cols, const double margin = 1, const double mean = 0) {
		std::mt19937_64 gen = seededRNG();
		const std::uniform_real_distribution<Scalar> dist(mean - margin, mean + margin);
		Vector ret(cols);
		for (size_t i = 0; i < cols; i++)
			ret(i)  = dist(gen);
		return ret;
	}
}

#endif