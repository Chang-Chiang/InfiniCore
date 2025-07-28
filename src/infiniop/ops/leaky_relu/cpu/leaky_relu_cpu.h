#ifndef __LEAKY_RELU_CPU_H__
#define __LEAKY_RELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(leaky_relu, cpu)

namespace op::leaky_relu::cpu {
typedef struct LeakyReluOp {
public:
    static constexpr size_t num_inputs = 1;
    float neagative_slope;

    LeakyReluOp(float neagative_slope_ = 0.01f) : neagative_slope(neagative_slope_) {}

    template <typename T>
    T operator()(const T &x) const {
        return x >= T(0) ? x : T(neagative_slope) * x;
    }
} LeakyReluOp;
} // namespace op::leaky_relu::cpu

#endif // __LEAKY_RELU_CPU_H__
