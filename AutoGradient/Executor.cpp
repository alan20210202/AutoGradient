//
// Created by Alan Ma on 2019/9/21.
//

#include "Executor.h"
#include <queue>
#include <unordered_set>
#include <sstream>
#include <typeinfo>

using namespace std;
using namespace autograd;

Executor::Executor(const OpPtr &result) {
    this->resultOp = result;
    unordered_map<OpPtr, size_t> inDegree;
    unordered_map<OpPtr, vector<OpPtr>> outOps;
    unordered_map<OpPtr, bool> visited;
    queue<OpPtr> qBFS, qTopoSort;
    qBFS.push(result);
    visited[result] = true;
    while (!qBFS.empty()) {
        auto u = qBFS.front(); qBFS.pop();
        auto inputs = u->inputs();
        if (inputs.empty())
            qTopoSort.push(u);
        inDegree[u] = inputs.size();
        for (const auto &v : inputs) {
            outOps[v].push_back(u);
            if (!visited[v]) {
                visited[v] = true;
                qBFS.push(v);
            }
        }
    }
    while (!qTopoSort.empty()) {
        auto u = qTopoSort.front(); qTopoSort.pop();
        this->order.push_back(u);
        for (const auto &v : outOps[u])
            if (--inDegree[v] == 0)
                qTopoSort.push(v);
    }
}

Value createOnesFor(const Value &v) {
    if (holds_alternative<Scalar>(v))
        return 1;
    if (holds_alternative<Matrix>(v)) {
        const Matrix &mat = get<Matrix>(v);
        return Matrix::Ones(mat.rows(), mat.cols());
    }
    Cube ret;
    for (const auto &mat : get<Cube>(v))
        ret.push_back(Matrix::Ones(mat.rows(), mat.cols()));
    return ret;
}

struct InvalidValueException : exception {
	InvalidValueException(const char *what) : exception(what) {}
};

// No more NaNs and Infs ... please!
void validateValue(const Value &v) {
	if (holds_alternative<Scalar>(v)) {
		const Scalar vx = get<Scalar>(v);
		if (isnan(vx))
			throw InvalidValueException("scalar nan");
		if (isinf(vx))
			throw InvalidValueException("scalar inf");
	} else if (holds_alternative<Matrix>(v)) {
		const Matrix& vx = get<Matrix>(v);
		if (vx.array().isNaN().count())
			throw InvalidValueException("matrix nan");
		if (vx.array().isInf().count())
			throw InvalidValueException("matrix inf");
	}
}

const Value &Executor::propagate(const bool withGradient) {
    for (const auto &v : order)
        lastValues[v] = v->eval(shared_from_this());
	// for (const auto& v : order)
	// 	validateValue(lastValues[v]);
    lastGrads.clear();
	if (!withGradient)
		return lastValues[resultOp];
    lastGrads[resultOp] = createOnesFor(valueOf(resultOp));
    for (auto itOrder = order.crbegin(); itOrder != order.crend(); ++itOrder) {
        const auto &curOp = *itOrder;
        auto inputs = curOp->inputs();
        auto gradInputs = curOp->diff(shared_from_this(), lastGrads[curOp]);
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto &inputOp = inputs[i];
            const auto &gradInput = gradInputs[i];
			// validateValue(gradInput);
            if (auto itGrad = lastGrads.find(inputOp); itGrad != lastGrads.end()) {
                if (holds_alternative<Scalar>(gradInput))
                    itGrad->second = get<Scalar>(itGrad->second) + get<Scalar>(gradInput);
                else if (holds_alternative<Matrix>(gradInput))
                    itGrad->second = get<Matrix>(itGrad->second) + get<Matrix>(gradInput);
                else {
                    const auto &cube = get<Cube>(gradInput);
                    auto &g = get<Cube>(itGrad->second);
                    for (size_t j = 0; j < cube.size(); j++)
                        g[j] += cube[j];
                }
			} else
				lastGrads.insert(make_pair(inputOp, gradInput));
        }
    }
	for (const auto &[op, grad] : lastGrads) {
		if (auto it = grads.find(op); it != grads.end()) {
			// TODO: Value accumulation seems to be commonly used, write a function 
			if (holds_alternative<Scalar>(grad))
				it->second = get<Scalar>(it->second) + get<Scalar>(grad);
			else if (holds_alternative<Matrix>(grad))
				it->second = get<Matrix>(it->second) + get<Matrix>(grad);
			else {
				const auto &cube = get<Cube>(grad);
				auto& g = get<Cube>(it->second);
				for (size_t i = 0; i < cube.size(); i++)
					g[i] += cube[i];
			}
		} else
			grads.insert(make_pair(op, grad));
	}
    return lastValues[resultOp];
}

string Executor::graph() const {
	stringstream ss;
	unordered_map<OpPtr, size_t> ids;
	auto nextId = 1;
	ss << "digraph g {" << endl;
	for (const auto &op : topoOrder()) {
		ss << "  " << nextId << "[label=\"" << typeid(*op).name() << "\"];" << endl;
		ids.insert(make_pair(op, nextId++));
	}
	for (const auto& op : topoOrder())
		for (const auto& in : op->inputs())
			ss << "  " << ids[in] << "->" << ids[op] << ";" << endl;
	ss << "}";
	return ss.str();
}

