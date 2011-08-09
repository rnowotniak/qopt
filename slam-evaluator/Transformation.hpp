#ifndef TRANSFORMATION_HPP
#define TRANSFORMATION_HPP

#include "Vec2.hpp"

struct Transformation
{
public:
    Transformation()
        : m_translation(0, 0), m_rotation(0)
    {}

    Transformation(const Vec2f & translation, float rotation)
        : m_translation(translation), m_rotation(rotation)
    {}
    
    Transformation(float dx, float dy, float rotation)
        : m_translation(dx, dy), m_rotation(rotation)
    {}

    Vec2f m_translation;
    float m_rotation;
};

#endif //TRANSFORMATION_HPP
