#ifndef AUTOGRADIENT_ADVANCEDOPS_H
#define AUTOGRADIENT_ADVANCEDOPS_H

#include <utility>
#include "Random.h"
#include "BasicOps.h"

// Here are some operations that may not be necessary (can be defined with other basic ops)
// but commonly used, so I put them here for better performance

namespace autograd {
	// dot(a, b) = sum(cwiseProduct(a, b))
	class DotOp : public Operator {
		BINARY_OP(DotOp)
		OVERRIDE_OUTPUT { return ValueType::Scalar; }
		OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)).cwiseProduct(std::get<Matrix>(V(rhs))).sum(); }
		OVERRIDE_DIFF {
			const Scalar vOutput = std::get<Scalar>(outputGrad);
			return { std::get<Matrix>(V(rhs)) * vOutput, std::get<Matrix>(V(lhs)) * vOutput };
		}
	};
	BINARY_OP_FUNC(dot, DotOp)

	// softmax(a) = exp(a) / (sum(exp(a)) + EPSILON)
	class SoftmaxOp : public Operator {
		const Scalar EPSILON = static_cast<Scalar>(1e-8);
		UNARY_OP(SoftmaxOp)
		OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL {
			const Array &x = std::get<Matrix>(V(operand));
			const Array &ex = (x - x.maxCoeff()).exp();
			return ex / (ex.sum() + EPSILON);
		}
		OVERRIDE_DIFF {
			const Array &x = std::get<Matrix>(V(operand));
			const Array &ex = (x - x.maxCoeff()).exp();
			const Array &r = ex.sum() - ex;
			const Matrix &d = ex / ((ex + r).square() + EPSILON) * r;
			return { d.cwiseProduct(std::get<Matrix>(outputGrad)) };
		}
	};
	UNARY_OP_FUNC(softmax, SoftmaxOp)
	
	// crossEntropy(a, b) = -dot(y, log(yHat)) - dot(1 - y, log(1 - yHat))
	class CrossEntropyOp : public Operator {
		const Scalar EPSILON = static_cast<Scalar>(1e-8);
		BINARY_OP(CrossEntropyOp)
		OVERRIDE_OUTPUT { return ValueType::Scalar; }
		OVERRIDE_EVAL{
			const Array &yHat = std::get<Matrix>(V(lhs));
			const Array &y = std::get<Matrix>(V(rhs));
			return -(y * (yHat + EPSILON).log()).sum() - ((1 - y) * (1 + EPSILON - yHat).log()).sum();
		}
		OVERRIDE_DIFF{
			const Scalar vOutput = std::get<Scalar>(outputGrad);
			const Array &yHat = std::get<Matrix>(V(lhs));
			const Array &y = std::get<Matrix>(V(rhs));
			return { ((1 - y) / (1 + EPSILON - yHat) - y / (yHat + EPSILON)) * vOutput,
					 vOutput * y * ((1 + EPSILON - yHat).log() - (yHat + EPSILON).log()) };
		}
	};
	BINARY_OP_FUNC_WITH_PARAM(crossEntropy, CrossEntropyOp, yHat, y)

	class DropoutOp : public Operator, public std::enable_shared_from_this<DropoutOp> {
		const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();
		bool training;
		OpPtr operand;
		Scalar dropRate;
		mutable std::mt19937_64 rng;
	public:
		DropoutOp(OpPtr x, Scalar dropRate, bool training = true)
			: training(training), operand(std::move(x)), dropRate(dropRate), rng(seededRNG()) {}
		void setTraining(const bool training) { this->training = training; }
		OVERRIDE_INPUTS { return { operand }; }
		OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL {
			if (!training)
				return (1 - dropRate) * std::get<Matrix>(V(operand));
			Matrix ret = std::get<Matrix>(V(operand));
			const std::uniform_real_distribution<Scalar> dist;
			for (auto i = 0; i < ret.rows(); i++)
				for (auto j = 0; j < ret.cols(); j++)
					if (dist(rng) < dropRate)
						ret(i, j) = ret(i, j) == 0 ? EPSILON : 0;
			// Why do we need to set 0 to EPSILON if a 0-value node is to be dropped?
			// Because IMO operators should be as stateless as possible (for states should be stored in Executor)
			// Then the diff() method has to rely on the difference of last output and last operand value
			// to determine which nodes were dropped in the last evaluation
			// If a node is already 0, then dropping it by setting it to 0 again will not make any "difference",
			// and therefore confuse the differentiator afterwards, so here we introduce a difference by setting
			// it to EPSILON
			return ret;
		}
		OVERRIDE_DIFF {
			if (!training)
				return { (1 - dropRate) * std::get<Matrix>(outputGrad) };
			const OpPtr pThis = std::static_pointer_cast<Operator>(
				std::const_pointer_cast<DropoutOp>(shared_from_this()));
			const Matrix &vThis = std::get<Matrix>(V(pThis));
			const Matrix& vOperand = std::get<Matrix>(V(operand));
			Matrix ret = std::get<Matrix>(outputGrad);
			// By comparing the last output and the last operand output we know which nodes were dropped
			for (auto i = 0; i < ret.rows(); i++)
				for (auto j = 0; j < ret.cols(); j++)
					ret(i, j) = vOperand(i, j) == vThis(i, j) ? ret(i, j) : 0;
			return { ret };
		}
	};
	inline OpPtr dropout(OpPtr operand, Scalar dropRate, bool training = true) {
		return std::static_pointer_cast<Operator>(
			std::make_shared<DropoutOp>(std::move(operand), dropRate, training));
	}
}

#endif