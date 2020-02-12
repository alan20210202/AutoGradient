//
// Created by Alan Ma on 2019/9/21.
//

#ifndef AUTOGRADIENT_EXECUTOR_H
#define AUTOGRADIENT_EXECUTOR_H

#include <unordered_map>
#include <string>
#include "Operator.h"
#include "Value.h"

namespace autograd {
    class Executor : public std::enable_shared_from_this<Executor> {
    protected:
	    virtual ~Executor() = default;
    private:
        std::vector<OpPtr> order;
        OpPtr resultOp;
        std::unordered_map<OpPtr, Value> lastValues;
        std::unordered_map<OpPtr, Value> lastGrads;
		std::unordered_map<OpPtr, Value> grads;
    public:
        explicit Executor(const OpPtr &result);
		void clearGradient() { grads.clear(); }
        const std::vector<OpPtr> &topoOrder() const { return order; }
        const Value &valueOf(const OpPtr &ptr) { return lastValues[ptr]; }
        const Value &gradientOf(const OpPtr &ptr) { return grads[ptr]; }
		const Value &lastGradientOf(const OpPtr &ptr) { return lastGrads[ptr]; }
        const Value &propagate(bool withGradient = true);
		std::string graph() const;
    };
}

#endif //AUTOGRADIENT_EXECUTOR_H
