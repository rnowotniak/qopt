#ifndef CACHEDKDTREE_VECTOR_HPP
#define CACHEDKDTREE_VECTOR_HPP

#include <memory>

namespace Utils
{

template <typename Vec2Type, int capacity = 1081>
class CachedKDTree_Vector
{
public:

    typedef Vec2Type * iterator;

    CachedKDTree_Vector()
        : m_begin(m_points), m_end(m_points)
    {
    }

    template <typename FwdIterator>
    void Fill(FwdIterator first, FwdIterator last)
    {
        resize(std::distance(first, last));
        Vec2Type * begin = m_begin;
        for(; first != last; ++first, ++begin)
        {
            begin->X() = first->X();
            begin->Y() = first->Y();
        }
    }

    Vec2Type * begin() { return m_begin; }
    Vec2Type * end() { return m_end; }

    void resize(size_t size) { m_end = m_begin + size; }
    void clear() { m_end = m_points; }
    size_t size() { return std::distance(m_begin, m_end); }

    void push_back(const Vec2Type & vec)
    {
        if(m_end == (m_points + capacity))
            return;
        *m_end++ = vec;
    }

private:
    Vec2Type m_points[capacity];

    Vec2Type * m_begin;
    Vec2Type * m_end;
};

}

#endif // CACHEDKDTREE_VECTOR_HPP
