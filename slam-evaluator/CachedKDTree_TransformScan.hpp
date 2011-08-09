#ifndef CACHEDKDTREE_TRANSFORMSCAN_HPP
#define CACHEDKDTREE_TRANSFORMSCAN_HPP

#include "Math.hpp"
#include "Vec2.hpp"

#include "Transformation.hpp"
#include "CachedKDTree_Vec2.hpp"

namespace Utils
{

class CachedKDTree_TransformScan
{
public:
    CachedKDTree_TransformScan(const Vec2f & translation, const float rotation)
        : m_translation(translation), m_sinRotation(std::sin(rotation)), m_cosRotation(std::cos(rotation))
    {}

    CachedKDTree_TransformScan(const Transformation & transformation)
        : m_translation(transformation.m_translation), m_sinRotation(std::sin(transformation.m_rotation)), m_cosRotation(std::cos(transformation.m_rotation))
    {}

    template <typename InIterator, typename OutIterator>
    OutIterator Transform(InIterator begin, InIterator end, OutIterator out)
    {
        for(; begin != end; ++begin)
        {
            out->X() = begin->X() * m_cosRotation - begin->Y() * m_sinRotation + m_translation.X();
            out->Y() = begin->X() * m_sinRotation + begin->Y() * m_cosRotation + m_translation.Y();
            out++;// = CachedKDTree_Vec2<float>(tmpx, tmpy, begin->treeNode);
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

}

#endif // CACHEDKDTREE_TRANSFORMSCAN_HPP
