#ifndef KDTREE_POOLALLOCATOR_HPP
#define KDTREE_POOLALLOCATOR_HPP

#include <list>

namespace Utils
{

template <typename T, int sliceSize = 1081>
class KDTree_PoolAllocator
{
public:
    KDTree_PoolAllocator()
    {
        AllocNewSlice();
    }

    ~KDTree_PoolAllocator()
    {
        for(typename std::list< T * >::iterator it = m_slices.begin(); it != m_slices.end(); ++it)
        {
            delete [] *it;
        }
    }

    T * GetNext()
    {
        if(m_endElement == m_currentElement)
        {
            AllocNewSlice();
        }
        return m_currentElement++;
    }

private:
    void AllocNewSlice()
    {
		T * slice = new T[sliceSize];
		m_currentElement = slice;
		m_endElement = m_currentElement + sliceSize;
		m_slices.push_back(slice);
    }
    
    std::list<T *> m_slices;
    
    T * m_currentElement;
    T * m_endElement;
};

}

#endif // KDTREE_POOLALLOCATOR_HPP
