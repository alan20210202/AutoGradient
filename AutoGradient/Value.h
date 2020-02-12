//
// Created by Alan Ma on 2019/9/21.
// Basic definitions
//

#ifndef AUTOGRADIENT_VALUE_H
#define AUTOGRADIENT_VALUE_H

#include "Eigen/Dense"
#include <variant>
#include <iostream>
namespace autograd {
    #ifndef AUTOGRADIENT_USE_FLOAT
    using Scalar = double;
    using Matrix = Eigen::MatrixXd;
	using Array = Eigen::ArrayXd;
    using Vector = Eigen::VectorXd;
    #else
    using Scalar = float;
    using Matrix = Eigen::MatrixXf;
	using Array = Eigen::ArrayXf;
    using Vector = Eigen::VectorXf;
    #endif
    // 3D Tensor, temporary implementation, the first dimension should be small
    // For CNN should be enough
    using Cube = std::vector<Matrix>;
    // The unified value type
    using Value = std::variant<Scalar, Matrix, Cube>;
    // This is used to identify the return type of the op
    enum class ValueType {
        Scalar,
        Matrix,
        Cube,
    };
    inline std::ostream &operator <<(std::ostream &out, const Value &v) {
        if (std::holds_alternative<Scalar>(v))
            return out << std::get<Scalar>(v);
        if (std::holds_alternative<Matrix>(v))
            return out << std::get<Matrix>(v);
        throw std::invalid_argument("unreachable code!");
    }
}
#endif //AUTOGRADIENT_VALUE_H
