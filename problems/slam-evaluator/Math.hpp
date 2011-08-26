#ifndef ICP_Math_HPP
#define ICP_Math_HPP

#include <cmath>

namespace ICP_Math
{

class Defines
{
public:
    static const double pi; // = 3.1415926535897932384626433832795;
    static const float pif; // = 3.14159265f;

    static const double pi2; // = 6.283185307179586476925286766559;
    static const float pi2f; // = 6.2831853f;
};

class Utils
{
public:
    static double ToRadians(double value)
    {
        return value * Defines::pi / 180.0;
    }

    static float ToRadians(float value)
    {
        return value * Defines::pif / 180.0f;
    }

    static double ToDegrees(double value)
    {
        return value * 180.0 / Defines::pi;
    }

    static float ToDegrees(float value)
    {
        return value * 180.0f / Defines::pif;
    }

    //template <typename T>
    //static inline float Abs(T value) { return std::abs(value); }

    //template <>
    //static inline float Abs<float>(float value)
    //{
    //    union
    //    {
    //        float f;
    //        int i;
    //    } temp;
    //    temp.f = value;
    //    temp.i = 0x7fffffff;
    //    return temp.f;
    //}
};

}

#endif // ICP_Math_HPP
