#ifndef TRANSFORMSCAN_HPP
#define TRANSFORMSCAN_HPP

#include "Math.hpp"
#include "Vec2.hpp"
#include "Transformation.hpp"


class TransformScan
{
public:
    TransformScan(const Vec2f & translation, const float rotation)
        : m_translation(translation), m_sinRotation(std::sin(rotation)), m_cosRotation(std::cos(rotation))
    {}

    TransformScan(const Transformation & transformation)
        : m_translation(transformation.m_translation), m_sinRotation(std::sin(transformation.m_rotation)), m_cosRotation(std::cos(transformation.m_rotation))
    {}

    template <typename InIterator, typename OutIterator>
    OutIterator Transform(InIterator begin, InIterator end, OutIterator out)
    {
        for(; begin != end; ++begin)
        {
            *out++ = begin->Rotate(m_sinRotation, m_cosRotation) + m_translation;
        }
        return out;
    }

    void SetTransformation(const Vec2f & translation, const float rotation)
    {
        m_translation = translation;
        m_sinRotation = std::sin(rotation);
        m_cosRotation = std::cos(rotation);
    }

private:
    Vec2f m_translation;
    float m_sinRotation;
    float m_cosRotation;
};

#endif // TRANSFORMSCAN_HPP
