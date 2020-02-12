//
// Created by Alan Ma on 2019/9/27.
//

#include "Optimizers.h"
#include <iostream>

using namespace autograd;
using namespace std;

void SGDOptimizer::update() {
    for (const auto &op : topoOrder()) {
        if (!op->updatable()) continue;
		if (op->outputType() == ValueType::Scalar)
			op->update(-rate * get<Scalar>(gradientOf(op)));
        else if (op->outputType() == ValueType::Matrix)
            op->update(-rate * get<Matrix>(gradientOf(op)));
    }
}

void AdamOptimizer::update() {
    updates++;
    const auto corr = static_cast<Scalar>(sqrt(1 - pow(beta2, updates)) / (1 - pow(beta1, updates)));
    for (const auto &op : topoOrder()) {
        if (!op->updatable()) continue;
        if (op->outputType() == ValueType::Scalar) {
            const auto grad = get<Scalar>(gradientOf(op));
            Scalar now1, now2;
			if (auto it = m1.find(op); it != m1.end())
				it->second = now1 = beta1 * get<Scalar>(it->second) + (1 - beta1) * grad;
			else
				m1.insert(make_pair(op, now1 = (1 - beta1) * grad));
			if (auto it = m2.find(op); it != m2.end())
				it->second = now2 = beta2 * get<Scalar>(it->second) + (1 - beta2) * grad * grad;
			else
				m2.insert(make_pair(op, now2 = (1 - beta2) * grad * grad));
            op->update(-alpha * corr * now1 / (sqrt(now2) + EPSILON));
        } else if (op->outputType() == ValueType::Matrix) {
            const Matrix &grad = get<Matrix>(gradientOf(op));
			if (auto it = m1.find(op); it != m1.end())
				it->second = beta1 * get<Matrix>(it->second) + (1 - beta1) * grad;
			else
				m1.insert(make_pair(op, (1 - beta1) * grad));
			if (auto it = m2.find(op); it != m2.end())
				it->second = beta2 * get<Matrix>(it->second) + (1 - beta2) * grad.cwiseProduct(grad);
			else
				m2.insert(make_pair(op, (1 - beta2) * grad.cwiseProduct(grad)));
			const auto& now1 = get<Matrix>(m1[op]).array();
			const auto& now2 = get<Matrix>(m2[op]).array();
            op->update(-alpha * corr * (now1 / (now2.sqrt() + EPSILON)).matrix());
        }
    }
}

void AdamWOptimizer::update() {
	for (const auto &op : topoOrder()) {
		if (!op->updatable()) continue;
		if (op->outputType() == ValueType::Scalar)
			op->update(-lambda * get<Scalar>(valueOf(op)));
		if (op->outputType() == ValueType::Matrix)
			op->update(-lambda * get<Matrix>(valueOf(op)));
	}
	AdamOptimizer::update();
}
