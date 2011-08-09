#ifndef KDTREE_CMP_HPP
#define KDTREE_CMP_HPP

namespace Utils
{

template <typename T>
bool KDTree_XComparer(const T & left, const T & right)
{
    return left.X() < right.X();
}

template <typename T>
bool KDTree_YComparer(const T & left, const T & right)
{
    return left.Y() < right.Y();
}

};

#endif // KDTREE_CMP_HPP
