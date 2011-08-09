#ifndef CACHEDKDTREE_VEC2_HPP
#define CACHEDKDTREE_VEC2_HPP

#include <cmath>
#include "CachedKDTree.hpp"
#include "Vec2.hpp"

namespace Utils
{

template <typename T>
class CachedKDTree_Vec2
{
public:
    typedef T MyItemType;
    typedef T value_type;
	
    explicit CachedKDTree_Vec2()
    {}

    CachedKDTree_Vec2(T x, T y)
        : m_x(x), m_y(y), treeNode(0)
    {}

    CachedKDTree_Vec2(const Vec2<T> & v)
        : m_x(v.X()), m_y(v.Y()), treeNode(0)
    {}
	
	CachedKDTree_Vec2(T x, T y, CachedKDTree_Node<CachedKDTree_Vec2<T> > * tn)
        : m_x(x), m_y(y), treeNode(tn)
    {}

    CachedKDTree_Vec2(const CachedKDTree_Vec2<T> & other)
		: m_x(other.m_x), m_y(other.m_y), treeNode(other.treeNode)
	{}

    CachedKDTree_Vec2<T> & operator=(const CachedKDTree_Vec2<T> & other)
    {
        m_x = other.m_x;
        m_y = other.m_y;
		treeNode = other.treeNode;
        return *this;
    }

    inline CachedKDTree_Vec2<T> operator+(const CachedKDTree_Vec2<T> & other)
    {
        return CachedKDTree_Vec2<T>(m_x + other.m_x, m_y + other.m_y);
    }

    inline CachedKDTree_Vec2<T> & operator+=(const CachedKDTree_Vec2<T> & other)
    {
        m_x += other.m_x;
        m_y += other.m_y;
        return *this;
    }

    inline CachedKDTree_Vec2<T> Rotate(const T sinRotation, const T cosRotation)
    {
        return CachedKDTree_Vec2<T>(m_x * cosRotation - m_y * sinRotation, m_x * sinRotation + m_y * cosRotation);
    }

    inline T Distance2(const CachedKDTree_Vec2<T> & other) const
    {
        T x = m_x - other.m_x;
        T y = m_y - other.m_y;
        return x * x + y * y;
    }

    inline T Distance(const CachedKDTree_Vec2<T> & other) const
    {
        T x = m_x - other.m_x;
        T y = m_y - other.m_y;
        return std::sqrt(x * x + y * y);
    }

    std::ostream & operator<<(std::ostream & out) const
    {
        out << "(" << m_x << ", " << m_y << ")";
        return out;
    }

    bool operator==(const CachedKDTree_Vec2<T> & other) const
    {
        return m_x == other.m_x && m_y == other.m_y;
    }

    bool operator!=(const CachedKDTree_Vec2<T> & other) const
    {
        return m_x != other.m_x || m_y != m_y;
    }

    inline T X() const { return m_x; }
    inline T Y() const { return m_y; }

    inline T & X() { return m_x; }
    inline T & Y() { return m_y; }


private:
    T m_x;
    T m_y;
public:
	CachedKDTree_Node<CachedKDTree_Vec2<T> > * treeNode;
};

}

template <typename T>
std::ostream & operator<<(std::ostream & out, const Utils::CachedKDTree_Vec2<T> & vec)
{
    vec << out;
    return out;
}

#endif // CACHEDKDTREE_VEC2_HPP
