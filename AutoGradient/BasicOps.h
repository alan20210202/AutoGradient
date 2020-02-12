//
// Created by Alan Ma on 2019/9/26.
//

#ifndef AUTOGRADIENT_BASICOPS_H
#define AUTOGRADIENT_BASICOPS_H

#include <cmath>
#include <iostream>

#include "Operator.h"
#include "Executor.h"

namespace autograd {

// Abbreviations of some boilerplate code
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define V(x) (env->valueOf(x))  
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define OVERRIDE_INPUTS std::vector<OpPtr> inputs() const override
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define OVERRIDE_OUTPUT ValueType outputType() const override
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define OVERRIDE_EVAL Value eval(std::shared_ptr<Executor> env) const override
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define OVERRIDE_DIFF std::vector<Value> diff(std::shared_ptr<Executor> env, const Value &outputGrad) const override


// Boilerplate code for binary operator
// Perhaps using CRTP is a better choice, but I'm happy with this one
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BINARY_OP(name) \
        OpPtr lhs, rhs; \
    public: \
        name(OpPtr lhs, OpPtr rhs) : lhs(std::move(lhs)), rhs(std::move(rhs)) {} \
        OVERRIDE_INPUTS { return { lhs, rhs }; };
// Boilerplate code for unary operator
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define UNARY_OP(name) \
        OpPtr operand; \
    public: \
        explicit name(OpPtr operand) : operand(std::move(operand)) {} \
        OVERRIDE_INPUTS { return { operand }; };
// Boilerplate code for input operator
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INPUT_OP(name, type, output) \
        type value; \
    public: \
        explicit name(const type &value) : value(value) {} \
        OVERRIDE_INPUTS { return {}; }; \
        OVERRIDE_OUTPUT { return output; }; \
        OVERRIDE_EVAL { return value; }; \
        OVERRIDE_DIFF { return {}; };
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define UNARY_FUNC(name, input, opname) \
    inline OpPtr name(input value) { \
        return std::static_pointer_cast<Operator>(std::make_shared<opname>(std::move(value))); \
    }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define UNARY_OP_FUNC(name, opname) UNARY_FUNC(name, OpPtr, opname)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BINARY_OP_FUNC_OVERLOAD_WITH_PARAM(name, arg1, arg2) \
    inline OpPtr name(Scalar arg1, OpPtr arg2) { return name(constant(arg1), std::move(arg2)); } \
    inline OpPtr name(OpPtr arg1, Scalar arg2) { return name(std::move(arg1), constant(arg2)); } \
    inline OpPtr name(const Matrix &arg1, OpPtr arg2) { return name(constant(arg1), std::move(arg2)); } \
    inline OpPtr name(OpPtr arg1, const Matrix &arg2) { return name(std::move(arg1), constant(arg2)); }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BINARY_OP_FUNC_WITH_PARAM(name, opname, arg1, arg2) \
    inline OpPtr name(OpPtr arg1, OpPtr arg2) { \
        return std::static_pointer_cast<Operator>(std::make_shared<opname>(std::move(arg1), std::move(arg2))); \
    } \
	BINARY_OP_FUNC_OVERLOAD_WITH_PARAM(name, arg1, arg2)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BINARY_OP_FUNC(name, opname) BINARY_OP_FUNC_WITH_PARAM(name, opname, a, b)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BINARY_OP_FUNC_OVERLOAD(name) BINARY_OP_FUNC_OVERLOAD_WITH_PARAM(name, a, b)

    // CLion parser really struggles on these, not sure if it's cause by my code or due to Eigen

    class ScalarConstOp : public Operator {
	    INPUT_OP(ScalarConstOp, Scalar, ValueType::Scalar)
		void set(const Scalar value) { this->value = value; }
    };
    UNARY_FUNC(constant, Scalar, ScalarConstOp)

    class MatrixConstOp : public Operator {
    	INPUT_OP(MatrixConstOp, Matrix, ValueType::Matrix)
		void set(const Matrix &value) { this->value = value; }
    };
    UNARY_FUNC(constant, Matrix, MatrixConstOp)

    class ScalarParamOp : public Operator {
        INPUT_OP(ScalarParamOp, Scalar, ValueType::Scalar)
        bool updatable() const override { return true; }
        void update(const Value &delta) override { value += std::get<Scalar>(delta); }
    };
    UNARY_FUNC(parameter, Scalar, ScalarParamOp)

    class MatrixParamOp : public Operator {
        INPUT_OP(MatrixParamOp, Matrix, ValueType::Matrix)
        bool updatable() const override { return true; }
        void update(const Value &delta) override { value += std::get<Matrix>(delta); }
    };
    UNARY_FUNC(parameter, Matrix, MatrixParamOp)

    class ScalarSumOp : public Operator {
        BINARY_OP(ScalarSumOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::get<Scalar>(V(lhs)) + std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF { return { outputGrad, outputGrad }; }
    };

    class ScalarDiffOp : public Operator {
        BINARY_OP(ScalarDiffOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::get<Scalar>(V(lhs)) - std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF { return { outputGrad, -std::get<Scalar>(outputGrad) }; }
    };

    class ScalarProductOp : public Operator {
        BINARY_OP(ScalarProductOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::get<Scalar>(V(lhs)) * std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF {
            return { std::get<Scalar>(V(rhs)) * std::get<Scalar>(outputGrad),
                        std::get<Scalar>(V(lhs)) * std::get<Scalar>(outputGrad) };
        }
    };

    class ScalarQuotientOp : public Operator {
        BINARY_OP(ScalarQuotientOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::get<Scalar>(V(lhs)) / std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF {
            const Scalar vLhs = std::get<Scalar>(V(lhs)), vRhs = std::get<Scalar>(V(rhs));
            const Scalar vOutput = std::get<Scalar>(outputGrad);
            return { vOutput / vRhs, vOutput * vLhs / (-vRhs * vRhs) };
        }
    };

    class MatrixSumOp : public Operator {
        BINARY_OP(MatrixSumOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)) + std::get<Matrix>(V(rhs)); }
        OVERRIDE_DIFF { return { outputGrad, outputGrad }; }
    };

    class MatrixDiffOp : public Operator {
        BINARY_OP(MatrixDiffOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)) - std::get<Matrix>(V(rhs)); }
        OVERRIDE_DIFF { return { outputGrad, -std::get<Matrix>(outputGrad) }; }
    };

    class MatrixProductOp : public Operator {
        BINARY_OP(MatrixProductOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL{
        	return std::get<Matrix>(V(lhs)) * std::get<Matrix>(V(rhs)); }
        OVERRIDE_DIFF {
	        const Matrix &vOutput = std::get<Matrix>(outputGrad);
	        return { vOutput * std::get<Matrix>(V(rhs)).transpose(),
                     std::get<Matrix>(V(lhs)).transpose() * vOutput };
        }
    };

    class MatrixScalarProductOp : public Operator {
        BINARY_OP(MatrixScalarProductOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return std::get<Scalar>(V(lhs)) * std::get<Matrix>(V(rhs)); }
        OVERRIDE_DIFF {
            const Matrix &vOutput = std::get<Matrix>(outputGrad);
            return { vOutput.cwiseProduct(std::get<Matrix>(V(rhs))).sum(),
                     std::get<Scalar>(V(lhs)) * vOutput };
        }
    };

    class MatrixScalarQuotientOp : public Operator {
        BINARY_OP(MatrixScalarQuotientOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)) / std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF {
            const Matrix &vOutput = std::get<Matrix>(outputGrad);
            const Scalar vRhs = std::get<Scalar>(V(rhs));
            return { vOutput / vRhs,
                     vOutput.cwiseProduct(std::get<Matrix>(V(lhs))).sum() / (-vRhs * vRhs) };
        }
    };

	class MatrixScalarSumOp : public Operator {
		BINARY_OP(MatrixScalarSumOp)
		OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)).array() + std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF { return { outputGrad, std::get<Matrix>(outputGrad).sum() }; }
	};

	class MatrixScalarDiffOp : public Operator {
		BINARY_OP(MatrixScalarDiffOp)
		OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)).array() - std::get<Scalar>(V(rhs)); }
        OVERRIDE_DIFF { return { outputGrad, -std::get<Matrix>(outputGrad).sum() }; }
	};

	class ScalarMatrixDiffOp : public Operator {
		BINARY_OP(ScalarMatrixDiffOp)
		OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL { return std::get<Scalar>(V(lhs)) - std::get<Matrix>(V(rhs)).array(); }
        OVERRIDE_DIFF { return { std::get<Matrix>(outputGrad).sum(), -std::get<Matrix>(outputGrad) }; }
	};

    class MatrixCWiseProductOp : public Operator {
        BINARY_OP(MatrixCWiseProductOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)).cwiseProduct(std::get<Matrix>(V(rhs))); };
        OVERRIDE_DIFF {
            const Matrix &vOutput = std::get<Matrix>(outputGrad);
            return { vOutput.cwiseProduct(std::get<Matrix>(V(rhs))),
                     vOutput.cwiseProduct(std::get<Matrix>(V(lhs))) };
        }
    };
    BINARY_OP_FUNC(cwiseProduct, MatrixCWiseProductOp)

    class MatrixCWiseQuotientOp : public Operator {
        BINARY_OP(MatrixCWiseQuotientOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(lhs)).cwiseQuotient(std::get<Matrix>(V(rhs))); };
        OVERRIDE_DIFF {
            const Matrix &vOutput = std::get<Matrix>(outputGrad);
            const Matrix &vRhs = std::get<Matrix>(V(rhs));
            return { vOutput.cwiseQuotient(vRhs),
                     -vOutput.cwiseProduct(std::get<Matrix>(V(lhs))).cwiseQuotient(vRhs).cwiseQuotient(vRhs) };
        }
    };
    BINARY_OP_FUNC(cwiseQuotient, MatrixCWiseQuotientOp)

    class ScalarPowOp : public Operator {
        BINARY_OP(ScalarPowOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::pow(std::get<Scalar>(V(lhs)), std::get<Scalar>(V(rhs))); };
        OVERRIDE_DIFF {
            const Scalar vLhs = std::get<Scalar>(V(lhs));
            const Scalar vRhs = std::get<Scalar>(V(rhs));
			const Scalar vOutput = std::get<Scalar>(outputGrad);
            return { vOutput * vRhs * std::pow(vLhs, vRhs - 1),
            		 vOutput * std::log(vLhs) * std::pow(vLhs, vRhs) };
        }
    };
    BINARY_OP_FUNC(pow, ScalarPowOp)

    class ScalarNegOp : public Operator {
        UNARY_OP(ScalarNegOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return -std::get<Scalar>(V(operand)); }
        OVERRIDE_DIFF { return { -std::get<Scalar>(outputGrad) }; }
    };

    class MatrixNegOp : public Operator {
        UNARY_OP(MatrixNegOp)
        OVERRIDE_OUTPUT { return ValueType::Matrix; }
        OVERRIDE_EVAL { return -std::get<Matrix>(V(operand)); }
        OVERRIDE_DIFF { return { -std::get<Matrix>(outputGrad) }; }
    };

    class MatrixCoefSumOp : public Operator {
        UNARY_OP(MatrixCoefSumOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(operand)).sum(); }
        OVERRIDE_DIFF {
            const Matrix &vOperand = std::get<Matrix>(V(operand));
			const Scalar vOutput = std::get<Scalar>(outputGrad);
			return { vOutput * Matrix::Ones(vOperand.rows(), vOperand.cols()) };
        }
    };
	UNARY_OP_FUNC(sum, MatrixCoefSumOp)

	class MatrixMaxOp : public Operator {
		UNARY_OP(MatrixMaxOp)
        OVERRIDE_OUTPUT { return ValueType::Scalar; }
        OVERRIDE_EVAL { return std::get<Matrix>(V(operand)).maxCoeff(); }
        OVERRIDE_DIFF {
            const Matrix &vOperand = std::get<Matrix>(V(operand));
			const Scalar vOutput = std::get<Scalar>(outputGrad);
			const Scalar max = vOperand.maxCoeff();
			Matrix ret(vOperand.rows(), vOperand.cols());
			for (auto i = 0; i < ret.rows(); i++)
				for (auto j = 0; j < ret.cols(); j++)
					ret(i, j) = vOperand(i, j) == max ? vOutput : 0;
			return { ret };
        }
	};
	UNARY_OP_FUNC(max, MatrixMaxOp)

	template <typename F>
	class FunctionApplyOp : public Operator {
		OpPtr x;
		const F &f;
	public:
		FunctionApplyOp(OpPtr x, const F &f) : x(std::move(x)), f(f) {}
		OVERRIDE_INPUTS { return { x }; }
		OVERRIDE_OUTPUT { return ValueType::Scalar; }
		OVERRIDE_EVAL { return f(std::get<Scalar>(V(x))); }
		OVERRIDE_DIFF { return { std::get<Scalar>(outputGrad) * f.d(std::get<Scalar>(V(x))) }; }
	};

	template <typename F>
	class FunctionBroadcastOp : public Operator {
		OpPtr x;
		const F &f;
	public:
		FunctionBroadcastOp(OpPtr x, const F &f) : x(std::move(x)), f(f) {}
		OVERRIDE_INPUTS { return { x }; }
		OVERRIDE_OUTPUT { return ValueType::Matrix; }
		OVERRIDE_EVAL {
			const Matrix &vx = std::get<Matrix>(V(x));
			Matrix ret(vx.rows(), vx.cols());
			for (auto i = 0; i < vx.rows(); i++)
				for (auto j = 0; j < vx.cols(); j++)
					ret(i, j) = f(vx(i, j));
			return ret;
		}
		OVERRIDE_DIFF{
			const Matrix &vx = std::get<Matrix>(V(x));
			Matrix ret(vx.rows(), vx.cols());
			for (auto i = 0; i < vx.rows(); i++)
				for (auto j = 0; j < vx.cols(); j++)
					ret(i, j) = f.d(vx(i, j));
			return { ret.cwiseProduct(std::get<Matrix>(outputGrad)) };
		}
	};

    [[noreturn]] inline void unreachable() { throw std::invalid_argument("unreachable or unimplemented code"); }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define OVERLOAD_BINARY_OP(op) \
    inline OpPtr operator op(Scalar a, OpPtr b) { return constant(a) op std::move(b); } \
    inline OpPtr operator op(OpPtr a, Scalar b) { return std::move(a) op constant(b); } \
    inline OpPtr operator op(const Matrix &a, OpPtr b) { return constant(a) op std::move(b); } \
    inline OpPtr operator op(OpPtr a, const Matrix &b) { return std::move(a) op constant(b); }

    inline OpPtr operator +(OpPtr a, OpPtr b) {
        const auto outA = a->outputType(), outB = b->outputType();
        if (outA == ValueType::Scalar && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<ScalarSumOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Matrix && outB == ValueType::Matrix)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixSumOp>(std::move(a), std::move(b)));
		if (outA == ValueType::Matrix && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixScalarSumOp>(std::move(a), std::move(b)));
		if (outA == ValueType::Scalar && outB == ValueType::Matrix)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixScalarSumOp>(std::move(b), std::move(a)));
        unreachable();
    }
    OVERLOAD_BINARY_OP(+)

    inline OpPtr operator -(OpPtr a, OpPtr b) {
        const auto outA = a->outputType(), outB = b->outputType();
        if (outA == ValueType::Scalar && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<ScalarDiffOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Matrix && outB == ValueType::Matrix)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixDiffOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Matrix && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixScalarDiffOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Scalar && outB == ValueType::Matrix)
            return std::static_pointer_cast<Operator>(std::make_shared<ScalarMatrixDiffOp>(std::move(a), std::move(b)));
        unreachable();
    }
    OVERLOAD_BINARY_OP(-)

    inline OpPtr operator *(OpPtr a, OpPtr b) {
        const auto outA = a->outputType(), outB = b->outputType();
        if (outA == ValueType::Scalar && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<ScalarProductOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Scalar && outB == ValueType::Matrix)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixScalarProductOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Matrix && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixScalarProductOp>(std::move(b), std::move(a)));
        if (outA == ValueType::Matrix && outB == ValueType::Matrix)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixProductOp>(std::move(a), std::move(b)));
        unreachable();
    }
    OVERLOAD_BINARY_OP(*)

    inline OpPtr operator /(OpPtr a, OpPtr b) {
        const auto outA = a->outputType(), outB = b->outputType();
        if (outA == ValueType::Scalar && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<ScalarQuotientOp>(std::move(a), std::move(b)));
        if (outA == ValueType::Matrix && outB == ValueType::Scalar)
            return std::static_pointer_cast<Operator>(std::make_shared<MatrixScalarQuotientOp>(std::move(a), std::move(b)));
        unreachable();
    }
    OVERLOAD_BINARY_OP(/)

    inline OpPtr operator -(OpPtr a) {
        switch (a->outputType()) {
            case ValueType::Scalar:
                return std::static_pointer_cast<Operator>(std::make_shared<ScalarNegOp>(std::move(a)));
            case ValueType::Matrix:
                return std::static_pointer_cast<Operator>(std::make_shared<MatrixNegOp>(std::move(a)));
            case ValueType::Cube:
                break;
        }
        unreachable();
    }

/*
#undef V
#undef OVERRIDE_INPUTS
#undef OVERRIDE_OUTPUT
#undef OVERRIDE_EVAL
#undef OVERRIDE_DIFF
#undef BINARY_OP
#undef UNARY_OP
#undef INPUT_OP
#undef UNARY_OP_FUNC
*/
}
#endif //AUTOGRADIENT_BASICOPS_H
