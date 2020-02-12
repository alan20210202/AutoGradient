//
// Created by Alan Ma on 2019/9/21.
// Basic operators

#ifndef AUTOGRADIENT_OPERATOR_H
#define AUTOGRADIENT_OPERATOR_H

#include <utility>
#include <vector>
#include <memory>
#include "Value.h"

namespace autograd {
    class Operator;
    class Executor;
    using OpPtr = std::shared_ptr<Operator>;

    class Operator {
    protected:
	    virtual ~Operator() = default;
    public:
        // Used by Executor to perform topo-sort and back-propagation
        virtual std::vector<OpPtr> inputs() const = 0;
        // Used by operator overloading to select appropriate operators
        virtual ValueType outputType() const = 0;
        // Evaluate
        virtual Value eval(std::shared_ptr<Executor> env) const = 0;
        // The order of returned gradients must match up with the result of inputs()
        virtual std::vector<Value> diff(std::shared_ptr<Executor> env, const Value &outputGrad) const = 0;
        // Does the op contains updatable parameters ?
        // This is for the optimizer
        virtual bool updatable() const { return false; }
        // If the op is updatable, call this
        virtual void update(const Value &delta) {}
    };
}
#endif //AUTOGRADIENT_OPERATOR_H
