
#ifndef ANAKIN_SABER_NORMAL_ACTIVATION_H
#define ANAKIN_SABER_NORMAL_ACTIVATION_H

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0
/*
template <typename Dtype>
inline Dtype InValidAct(Dtype a) {
    CHECK_EQ(0,1)<<"InValidAct";
}

template <typename Dtype>
inline Dtype Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
}

template <typename Dtype>
inline Dtype Sigmoid_fluid(const Dtype a) {
    const Dtype min = SIGMOID_THRESHOLD_MIN;
    const Dtype max = SIGMOID_THRESHOLD_MAX;
    Dtype tmp = (a < min) ? min : ((a > max) ? max : a);
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-tmp));
}

template <typename Dtype>
inline Dtype Tanh_fluid(const Dtype a) {
    Dtype tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template <typename Dtype>
inline Dtype Tanh(const Dtype a) {
    Dtype tmp = -2.0 * a;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template <typename Dtype>
inline Dtype Relu(const Dtype a) {
    return a > static_cast<Dtype>(0.0) ? a : static_cast<Dtype>(0.0);
}

template <typename Dtype>
inline Dtype Identity(const Dtype a) {
    return a;
}
*/

inline float InValidAct(float a) {
            CHECK_EQ(0,1)<<"InValidAct";
}


inline float Sigmoid(const float a) {
    return static_cast<float>(1.0) / (static_cast<float>(1.0) + exp(-a));
}


inline float Sigmoid_fluid(const float a) {
    const float min = SIGMOID_THRESHOLD_MIN;
    const float max = SIGMOID_THRESHOLD_MAX;
    float tmp = (a < min) ? min : ((a > max) ? max : a);
    return static_cast<float>(1.0) / (static_cast<float>(1.0) + exp(-tmp));
}


inline float Tanh_fluid(const float a) {
    float tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

inline float Tanh(const float a) {
    float tmp = -2.0 * a;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

inline float Relu(const float a) {
    return a > static_cast<float>(0.0) ? a : static_cast<float>(0.0);
}


inline float Identity(const float a) {
    return a;
}
static float (*act_func[])(float)= {&InValidAct, &Sigmoid, &Relu, &Tanh, &InValidAct, \
                                        & InValidAct, &Identity, &Sigmoid_fluid, &Tanh_fluid
};
#endif //ANAKIN_SABER_NORMAL_ACTIVATION_H
