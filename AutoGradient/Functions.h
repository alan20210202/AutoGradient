#ifndef AUTOGRADIENT_FUNCTIONS_H
#define AUTOGRADIENT_FUNCTIONS_H

#include <cmath>

#include "Value.h"
#include "BasicOps.h"

namespace autograd {
	// Ahh! operator() cannot be non-member, so CRTP trick no longer works
	// This macro makes custom function callable by OpPtr arguments
#define FUNCTION_CALL_OVERLOAD(name) \
	OpPtr operator ()(OpPtr x) const { \
		if (x->outputType() == ValueType::Scalar) \
			return std::static_pointer_cast<Operator>(std::make_shared<FunctionApplyOp<name>>(std::move(x), *this)); \
		if (x->outputType() == ValueType::Matrix) \
			return std::static_pointer_cast<Operator>(std::make_shared<FunctionBroadcastOp<name>>(std::move(x), *this)); \
		unreachable(); \
	}

	inline struct SinFunction {
		FUNCTION_CALL_OVERLOAD(SinFunction)
		Scalar operator ()(const Scalar x) const { return std::sin(x); }
		Scalar d(const Scalar x) const { return std::cos(x); }
	} sin;

	inline struct CosFunction {
		FUNCTION_CALL_OVERLOAD(CosFunction)
		Scalar operator ()(const Scalar x) const { return std::cos(x); }
		Scalar d(const Scalar x) const { return -std::sin(x); }
	} cos;

	inline struct LogFunction {
		const Scalar EPSILON = 1e-8;
		FUNCTION_CALL_OVERLOAD(LogFunction)
		Scalar operator ()(const Scalar x) const { return std::log(x + EPSILON); }
		Scalar d(const Scalar x) const { return 1 / (x + EPSILON); }
	} log;

	inline struct ExpFunction {
		FUNCTION_CALL_OVERLOAD(ExpFunction)
		Scalar operator ()(const Scalar x) const { return std::exp(x); }
		Scalar d(const Scalar x) const { return std::exp(x); }
	} exp;

	inline struct TanhFunction {
		FUNCTION_CALL_OVERLOAD(TanhFunction)
		Scalar operator ()(const Scalar x) const { return std::tanh(x); }
		Scalar d(const Scalar x) const { return 1 - std::tanh(x) * std::tanh(x); }
	} tanh;

	inline struct SigmoidFunction {
		FUNCTION_CALL_OVERLOAD(SigmoidFunction)
		Scalar operator ()(const Scalar x) const { return 1 / (1 + std::exp(-x)); }
		Scalar d(const Scalar x) const { return (*this)(x) * (1 - (*this)(x)); }
	} sigmoid;

	inline struct LReLUFunction {
		FUNCTION_CALL_OVERLOAD(LReLUFunction)
		Scalar operator ()(const Scalar x) const { return x > 0 ? x : 0.01 * x; }
		Scalar d(const Scalar x) const { return x > 0 ? 1 : 0.01; }
	} lrelu;

	inline struct MishFunction {
		FUNCTION_CALL_OVERLOAD(MishFunction)
		Scalar operator ()(const Scalar x) const { return x * std::tanh(std::log(std::exp(x) + 1)); }
		Scalar d(const Scalar x) const {
			const Scalar a = std::exp(x);
			const Scalar b = std::log(a + 1);
			const Scalar c = std::tanh(b);
			return c + x * a / (a + 1) * (1 - c * c);
		}
	} mish;
}

#endif