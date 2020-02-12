#ifndef AUTOGRADIENT_RANDOM_H
#define AUTOGRADIENT_RANDOM_H

#include <random>
#include <chrono>

namespace autograd {
	inline std::mt19937_64 seededRNG() {
		return std::mt19937_64(
			std::chrono::high_resolution_clock::now()
			.time_since_epoch().count()
		);
	}
}

#endif