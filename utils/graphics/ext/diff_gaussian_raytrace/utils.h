#ifndef __UTILS_H__
#define __UTILS_H__

#ifdef __CUDACC__
#define DEVICE __device__ __host__
#else  // CUDACC
#define DEVICE 
#endif // CUDACC
template<typename T>
struct vec2 {
    T x;
    T y;
};

template<typename T>
DEVICE inline T vec2_cross(const vec2<T>& lhs, const vec2<T>& rhs) {
    return (lhs.x*rhs.y - rhs.x*lhs.y);
};

template<typename T>
DEVICE inline vec2<T> make_vec2(const T& x, const T& y){
    vec2<T> v2;
    v2.x = x;
    v2.y = y;
    return v2;
};

template<typename T>
struct vec3 {
    T x, y, z;
    // inline vec3(const T x, const T y, const T z):x(x), y(y), z(z){};
    // inline vec3():vec3{0, 0, 0}{};


    DEVICE inline vec3<T> cross(const vec3<T>& rhs) const {
        vec3<T> v3;
        v3.x =   y*rhs.z - z*rhs.y;
        v3.y = -(x*rhs.z - z*rhs.x);
        v3.z =   x*rhs.y - y*rhs.x;
        return v3;
    };
    DEVICE inline T dot(const vec3<T>& rhs) const{
        T ret = static_cast<T>(0);
        ret += x * rhs.x;
        ret += y * rhs.y;
        ret += z * rhs.z;
        return ret;
    };
    DEVICE inline vec3<T> operator-(const vec3<T>& rhs) const{
        // const auto lhs = *this;
        vec3<T> v3;
        v3.x = x - rhs.x;
        v3.y = y - rhs.y;
        v3.z = z - rhs.z;
        return v3;
    };
    DEVICE inline vec3<T> operator+(const vec3<T>& rhs) const{
        vec3<T> v3;
        v3.x = x + rhs.x;
        v3.y = y + rhs.y;
        v3.z = z + rhs.z;
        return v3;
    };
    DEVICE inline vec3<T> operator*(T rhs) const{
        // const auto lhs = *this;
        vec3<T> v3;
        v3.x = x * rhs;
        v3.y = y * rhs;
        v3.z = z * rhs;
        return v3;
    };
};
template<typename T>
DEVICE inline vec3<T> make_vec3(const T& x, const T& y, const T& z){
    vec3<T> v3;
    v3.x = x;
    v3.y = y;
    v3.z = z;
    return v3;
};
template<typename T>
DEVICE inline vec3<T> vec3_cross(const vec3<T>& lhs, const vec3<T>& rhs) {
    vec3<T> v3;
    v3.x =   lhs.y*rhs.z - lhs.z*rhs.y;
    v3.y = -(lhs.x*rhs.z - lhs.z*rhs.x);
    v3.z =   lhs.x*rhs.y - lhs.y*rhs.x;
    return v3;
};
template<typename T>
DEVICE inline T vec3_dot(const vec3<T>& lhs, const vec3<T>& rhs) {
    T ret = static_cast<T>(0);
    ret += lhs.x * rhs.x;
    ret += lhs.y * rhs.y;
    ret += lhs.z * rhs.z;
    return ret;
};
template<typename T>
DEVICE inline vec3<T> vec3_sub(const vec3<T>& lhs, const vec3<T>& rhs) {
    // const auto lhs = *this;
    vec3<T> v3;
    v3.x = lhs.x - rhs.x;
    v3.y = lhs.y - rhs.y;
    v3.z = lhs.z - rhs.z;
    return v3;
};
template<typename T>
DEVICE inline vec3<T> vec3_add(const vec3<T>& lhs, const vec3<T>& rhs) {
    vec3<T> v3;
    v3.x = lhs.x + rhs.x;
    v3.y = lhs.y + rhs.y;
    v3.z = lhs.z + rhs.z;
    return v3;
};
template<typename T>
DEVICE inline vec3<T> vec3_mul(const vec3<T>& lhs, T rhs) {
    // const auto lhs = *this;
    vec3<T> v3;
    v3.x = lhs.x * rhs;
    v3.y = lhs.y * rhs;
    v3.z = lhs.z * rhs;
    return v3;
};
#endif