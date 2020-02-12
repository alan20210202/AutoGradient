//
// Created by Alan Ma on 2019/9/27.
//

#ifndef AUTOGRADIENT_OPTIMIZERS_H
#define AUTOGRADIENT_OPTIMIZERS_H

#include "Executor.h"

namespace autograd {
    class Optimizer : public Executor {
    public:
        explicit Optimizer(const OpPtr &resultOp) : Executor(resultOp) {}
        virtual void update() = 0;
    };

    class SGDOptimizer : public Optimizer {
        double rate;
    public:
        SGDOptimizer(const OpPtr &resultOp, double rate) : Optimizer(resultOp), rate(rate) {}
        void update() override;
    };

    class AdamOptimizer : public Optimizer {
        const Scalar EPSILON = static_cast<Scalar>(1e-8);
        Scalar alpha, beta1, beta2;
		size_t updates;
        std::unordered_map<OpPtr, Value> m1, m2;
    public:
        explicit AdamOptimizer(const OpPtr &resultOp, Scalar alpha = 0.001, Scalar beta1 = 0.9, Scalar beta2 = 0.999)
            : Optimizer(resultOp), alpha(alpha), beta1(beta1), beta2(beta2), updates(0) {}
        void update() override;
    };
	
	class AdamWOptimizer : public AdamOptimizer {
        const Scalar EPSILON = static_cast<Scalar>(1e-8);
        Scalar lambda;
    public:
        AdamWOptimizer(const OpPtr &resultOp, Scalar lambda, Scalar alpha = 0.001, Scalar beta1 = 0.9, Scalar beta2 = 0.999)
            : lambda(lambda), AdamOptimizer(resultOp, alpha, beta1, beta2) {}
        void update() override;
    };
}

#endif //AUTOGRADIENT_OPTIMIZERS_H
