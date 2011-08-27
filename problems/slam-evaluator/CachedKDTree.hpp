#ifndef CACHEDKDTREE_HPP
#define CACHEDKDTREE_HPP

#include <iostream>
#include <limits>
#include <algorithm>

#include "KDTree_PoolAllocator.hpp"
#include "KDTree_Cmp.hpp"

#undef min
#undef max

namespace Utils
{

#define CAHCED_KDTREE_BEST_ABS(x) ICP_Math::Utils::Abs(x)

//#define CAHCED_KDTREE_BEST_ABS(x) std::abs(x)


#define CACHED_KDTREE_SAFE_INFINITE 1.e10f

template <int axis>
struct CachedKDTree_Traits
{
public:
    template <typename Vec2Type>
    static float GetValue(const Vec2Type & v) { if(axis == 0) return v.X(); else if(axis == 1) return v.Y(); else return 0.0f; }

    template <typename NodeType>
    static float GetMin(const NodeType * node) { return 0.0f; }

    template <typename NodeType>
    static float GetMax(const NodeType * node) { return 0.0f; }
    
    template <typename NodeType>
    static void SetMin(NodeType * node, float value) { }

    template <typename NodeType>
    static void SetMax(NodeType * node, float value) { }
};

template <>
struct CachedKDTree_Traits<0>
{
public:
    template <typename Vec2Type>
    static float GetValue(const Vec2Type & v) { return v.X(); }

    template <typename NodeType>
    static float GetMin(const NodeType * node) { return node->minX; }

    template <typename NodeType>
    static float GetMax(const NodeType * node) { return node->maxX; }

    template <typename NodeType>
    static void SetMin(NodeType * node, float value) { node->minX = value; }

    template <typename NodeType>
    static void SetMax(NodeType * node, float value) { node->maxX = value; }
};

template <>
struct CachedKDTree_Traits<1>
{
public:
    template <typename Vec2Type>
    static float GetValue(const Vec2Type & v) { return v.Y(); }

    template <typename NodeType>
    static float GetMin(const NodeType * node) { return node->minY; }

    template <typename NodeType>
    static float GetMax(const NodeType * node) { return node->maxY; }

    template <typename NodeType>
    static void SetMin(NodeType * node, float value) { node->minY = value; }

    template <typename NodeType>
    static void SetMax(NodeType * node, float value) { node->maxY = value; }
};


template <typename Vec2Type>
class CachedKDTree_Node
{
    typedef CachedKDTree_Node<Vec2Type> Me;
public:
    CachedKDTree_Node()
    {};

public:
    Me * child[2];
    Me * parrent;
    
    typename Vec2Type::value_type value;

    Vec2Type element;
    bool isLeaf;

    int axis;
    int childSide;
    
    typename Vec2Type::value_type minX;
    typename Vec2Type::value_type minY;
    typename Vec2Type::value_type maxX;
    typename Vec2Type::value_type maxY;
};

template <
	typename Vec2Type,
	int AllocatorSliceSize = 1081,
	typename Allocator = Utils::KDTree_PoolAllocator<
		CachedKDTree_Node<Vec2Type>,
		AllocatorSliceSize > >
class CachedKDTree
{


    typedef CachedKDTree_Node<Vec2Type> NodeType;

public:
    typedef Vec2Type value_type;

    template <typename RandIterator>
	CachedKDTree(RandIterator first, RandIterator last)
	{
        root = BuildTreeRec(first, last, 0, 0, 0, -CACHED_KDTREE_SAFE_INFINITE, -CACHED_KDTREE_SAFE_INFINITE, CACHED_KDTREE_SAFE_INFINITE, CACHED_KDTREE_SAFE_INFINITE);
	}

    void FindCachedNN(Vec2Type & point, Vec2Type & nn, float & distance2)
    {
        distance2 = point.Distance2(point.treeNode->element);
        nn = point.treeNode->element;

        if(point.treeNode->axis == 0)
            FindCachedNNRec<1>(point.treeNode, point, nn, distance2);
		else
            FindCachedNNRec<0>(point.treeNode, point, nn, distance2);
    }

    void FindNN(Vec2Type & point, Vec2Type & nn, float & distance2)
    {
        distance2 = std::numeric_limits<float>::max();
        FindNNRec<0>(root, point, nn, distance2);
    }

    void Insert(Vec2Type & point)
    {
        InsertRec<0>(root, point);
    }

    template <typename FwdIterator>
    void Insert(FwdIterator first, FwdIterator last)
    {
        for(; first != last; ++first)
            Insert(*first);
    }

    void Write(std::ostream & out) const
    {
        WriteRec(root, out);
    }

private:
    template <typename RandIterator>
    NodeType * BuildTreeRec(RandIterator first, RandIterator last, NodeType * parrent, int axis, int childSide, float minX, float minY, float maxX, float maxY)
    {
        if(first == last)
            return 0;
        
        CachedKDTree_Node<Vec2Type> * node = m_poolAllocator.GetNext();
        
        if((first + 1) == last)
        {
            node->isLeaf = true;
            node->element = *first;
        }
        else
        {

            RandIterator center = first + std::distance(first, last) / 2;
            float leftMax = -std::numeric_limits<float>::max();

            if(axis == 0) // X;
            {
                std::nth_element(first, center, last, KDTree_XComparer<Vec2Type>);

				for(RandIterator temp = first; temp != center; ++temp)
					if(temp->X() > leftMax)
						leftMax = temp->X();
                
                node->value = (center->X() + leftMax) / 2.0f;
            
                node->child[0] = BuildTreeRec(first, center, node, 1, 0, minX, minY, center->X(), maxY);
                node->child[1] = BuildTreeRec(center, last, node, 1, 1, leftMax, minY, maxX, maxY);
                //node->child[0] = BuildTreeRec(first, center, node, 1, 0, minX, minY, node->value, maxY);
                //node->child[1] = BuildTreeRec(center, last, node, 1, 1, node->value, minY, maxX, maxY);
            }
            else
            {
                std::nth_element(first, center, last, KDTree_YComparer<Vec2Type>);

				for(RandIterator temp = first; temp != center; ++temp)
					if(temp->Y() > leftMax)
						leftMax = temp->Y();
                
                node->value = (center->Y() + leftMax) / 2.0f;
                
                node->child[0] = BuildTreeRec(first, center, node, 0, 0, minX, minY, maxX, center->Y());
                node->child[1] = BuildTreeRec(center, last, node, 0, 1, minX, leftMax, maxX, maxY);
                //node->child[0] = BuildTreeRec(first, center, node, 0, 0, minX, minY, maxX, node->value);
                //node->child[1] = BuildTreeRec(center, last, node, 0, 1, minX, node->value, maxX, maxY);
            }
            node->isLeaf = false;
        }
        
        node->parrent = parrent;
        node->minX = minX;
        node->minY = minY;
        node->maxX = maxX;
        node->maxY = maxY;
        node->axis = axis;
        node->childSide = childSide;
        
        return node;
    }

    template <int axis>
    void FindNNRec(NodeType * node, Vec2Type & point, Vec2Type & nn, float & distance2)
    {
        if(node == 0)
            return;

        if(node->isLeaf == true)
        {
            //float d = point.Distance(node->element);
            
            float d = point.Distance2(node->element);
            
            if(d < distance2)
            {
                distance2 = d;
                nn = node->element;
                point.treeNode = node;
            }
        }
        else
        {
            //float axisDistance = std::abs(CachedKDTree_Traits<axis>::GetValue(point) - node->value);
            
            float axisDistance = CachedKDTree_Traits<axis>::GetValue(point) - node->value;
            axisDistance *= axisDistance;
            
            if(CachedKDTree_Traits<axis>::GetValue(point) < node->value) // go left
            {
                FindNNRec<(axis + 1) & 1>(node->child[0], point, nn, distance2);
                
                if(distance2 > axisDistance)
                {
                    FindNNRec<(axis + 1) & 1>(node->child[1], point, nn, distance2);
                }
            }
            else
            {
                FindNNRec<(axis + 1) & 1>(node->child[1], point, nn, distance2);
            
                if(distance2 > axisDistance)
                {
                    FindNNRec<(axis + 1) & 1>(node->child[0], point, nn, distance2);
                }
            }
        }
    }

    template <int axis>
    void FindCachedNNRec(NodeType * node, Vec2Type & point, Vec2Type & nn, float & distance2)
    {
        float tempDistance;

		//tempDistance = CachedKDTree_Traits<axis>::GetValue(point) - CachedKDTree_Traits<axis>::GetMax(node);
  //      tempDistance *= tempDistance;

  //      if(tempDistance > distance2) // should check another child of parrent
  //      {
  //          tempDistance = CachedKDTree_Traits<axis>::GetValue(point) - CachedKDTree_Traits<axis>::GetMin(node);
  //          tempDistance *= tempDistance;

  //          if(tempDistance > distance2)
  //          {
  //              tempDistance = CachedKDTree_Traits<(axis + 1) & 1>::GetValue(point) - CachedKDTree_Traits<(axis + 1) & 1>::GetMin(node);
  //              tempDistance *= tempDistance;
  //              
  //              if(tempDistance > distance2)
  //              {
  //                  tempDistance = CachedKDTree_Traits<(axis + 1) & 1>::GetValue(point) - CachedKDTree_Traits<(axis + 1) & 1>::GetMax(node);
  //                  tempDistance *= tempDistance;
  //                  
  //                  if(tempDistance > distance2)
  //                      return;
  //              }
  //          }
		//}

  //      FindNNRec<(axis + 1) & 1>(node->parrent->child[(node->childSide + 1) & 1], point, nn, distance2);
  //      FindCachedNNRec<(axis + 1) & 1>(node->parrent, point, nn, distance2);


		if(node->childSide == 0) // left
        {
            tempDistance = CachedKDTree_Traits<axis>::GetValue(point) - CachedKDTree_Traits<axis>::GetMax(node);
            tempDistance *= tempDistance;

            if(tempDistance <= distance2) // should check another child of parrent
            {
                FindNNRec<(axis + 1) & 1>(node->parrent->child[1], point, nn, distance2);
                FindCachedNNRec<(axis + 1) & 1>(node->parrent, point, nn, distance2);
            }
            else
            {
                tempDistance = CachedKDTree_Traits<axis>::GetValue(point) - CachedKDTree_Traits<axis>::GetMin(node);
                tempDistance *= tempDistance;

                if(tempDistance > distance2)
                {
                    tempDistance = CachedKDTree_Traits<(axis + 1) & 1>::GetValue(point) - CachedKDTree_Traits<(axis + 1) & 1>::GetMin(node);
                    tempDistance *= tempDistance;
                    
                    if(tempDistance > distance2)
                    {
                        tempDistance = CachedKDTree_Traits<(axis + 1) & 1>::GetValue(point) - CachedKDTree_Traits<(axis + 1) & 1>::GetMax(node);
                        tempDistance *= tempDistance;
                        
                        if(tempDistance > distance2)
                            return;
                    }
                }

                // other bounds exceeded... must look up

                FindCachedNNRec<(axis + 1) & 1>(node->parrent, point, nn, distance2);
            }
        }
        else  // right
        {
            tempDistance = CachedKDTree_Traits<axis>::GetValue(point) - CachedKDTree_Traits<axis>::GetMin(node);
            tempDistance *= tempDistance;

            if(tempDistance <= distance2) // should check another child of parrent
            {
                FindNNRec<(axis + 1) & 1>(node->parrent->child[0], point, nn, distance2);
                FindCachedNNRec<(axis + 1) & 1>(node->parrent, point, nn, distance2);
            }
            else
            {
                tempDistance = CachedKDTree_Traits<axis>::GetValue(point) - CachedKDTree_Traits<axis>::GetMax(node);
                tempDistance *= tempDistance;

                if(tempDistance > distance2)
                {
                    tempDistance = CachedKDTree_Traits<(axis + 1) & 1>::GetValue(point) - CachedKDTree_Traits<(axis + 1) & 1>::GetMin(node);
                    tempDistance *= tempDistance;
                    if(tempDistance > distance2)
                    {
                        tempDistance = CachedKDTree_Traits<(axis + 1) & 1>::GetValue(point) - CachedKDTree_Traits<(axis + 1) & 1>::GetMax(node);
                        tempDistance *= tempDistance;
                        
                        if(tempDistance > distance2)
                            return;
                    }
                }

                // other bounds exceeded... must look up

                FindCachedNNRec<(axis + 1) & 1>(node->parrent, point, nn, distance2);
            }
        }
    }

    template <int axis>
    void InsertRec(NodeType * node, Vec2Type & point)
    {
        if(node->isLeaf == true)
        {
            node->isLeaf = false;
            node->value = (CachedKDTree_Traits<axis>::GetValue(point) + CachedKDTree_Traits<axis>::GetValue(node->element)) / 2;
            
            NodeType * left = m_poolAllocator.GetNext(); // left
            NodeType * right = m_poolAllocator.GetNext(); // right
            
            left->isLeaf = true;
            right->isLeaf = true;

            left->parrent = node;
            right->parrent = node;

            left->axis = (axis + 1) & 1;
            right->axis = left->axis;

            if(CachedKDTree_Traits<axis>::GetValue(point) < node->value) // point -> left, element -> right
            {
                left->element = point;
                right->element = node->element;
            }
            else
            {
                left->element = node->element;
                right->element = point;
            }

            CachedKDTree_Traits<axis>::SetMin(left, CachedKDTree_Traits<axis>::GetMin(node));
            CachedKDTree_Traits<axis>::SetMax(left, CachedKDTree_Traits<axis>::GetMax(node));
            CachedKDTree_Traits<(axis + 1) & 1>::SetMin(left, CachedKDTree_Traits<(axis + 1) & 1>::GetMin(node));
            CachedKDTree_Traits<(axis + 1) & 1>::SetMax(left, CachedKDTree_Traits<(axis + 1) & 1>::GetValue(right->element));
            
            CachedKDTree_Traits<axis>::SetMin(right, CachedKDTree_Traits<axis>::GetMin(node));
            CachedKDTree_Traits<axis>::SetMax(right, CachedKDTree_Traits<axis>::GetMax(node));
            CachedKDTree_Traits<(axis + 1) & 1>::SetMin(right, CachedKDTree_Traits<(axis + 1) & 1>::GetValue(left->element));
            CachedKDTree_Traits<(axis + 1) & 1>::SetMax(right, CachedKDTree_Traits<(axis + 1) & 1>::GetMax(node));

            //CachedKDTree_Traits<axis>::SetMin(left, CachedKDTree_Traits<axis>::GetMin(node));
            //CachedKDTree_Traits<axis>::SetMax(left, CachedKDTree_Traits<axis>::GetValue(right->element));
            //CachedKDTree_Traits<(axis + 1) & 1>::SetMin(left, CachedKDTree_Traits<(axis + 1) & 1>::GetMin(node));
            //CachedKDTree_Traits<(axis + 1) & 1>::SetMax(left, CachedKDTree_Traits<(axis + 1) & 1>::GetMax(node));
            //
            //CachedKDTree_Traits<axis>::SetMin(right, CachedKDTree_Traits<axis>::GetValue(left->element));
            //CachedKDTree_Traits<axis>::SetMax(right, CachedKDTree_Traits<axis>::GetMax(node));
            //CachedKDTree_Traits<(axis + 1) & 1>::SetMin(right, CachedKDTree_Traits<(axis + 1) & 1>::GetMin(node));
            //CachedKDTree_Traits<(axis + 1) & 1>::SetMax(right, CachedKDTree_Traits<(axis + 1) & 1>::GetMax(node));

            node->child[0] = left;
            node->child[1] = right;
        }
        else
        {
            if(CachedKDTree_Traits<axis>::GetValue(point) < node->value) // go left, update right
            {
                //if(CachedKDTree_Traits<axis>::GetMax(node->child[1]) > CachedKDTree_Traits<axis>::GetValue(point)) // bounding box should be fited to new dimension
                {
                    UpdateRecMax<axis>(node->child[1], CachedKDTree_Traits<axis>::GetValue(point));
                }
                InsertRec<(axis + 1) & 1>(node->child[0], point);
            }
            else // go right, update left
            {
                //if(CachedKDTree_Traits<axis>::GetMin(node->child[0]) < CachedKDTree_Traits<axis>::GetValue(point)) // bounding box should be fited to new dimension
                {
                    UpdateRecMin<axis>(node->child[0], CachedKDTree_Traits<axis>::GetValue(point));
                }
                InsertRec<(axis + 1) & 1>(node->child[1], point);
            }
        }
    }

    template <int axis>
    void UpdateRecMax(NodeType * node, float value)
    {
        if(node->isLeaf)
        {
            if(CachedKDTree_Traits<axis>::GetMax(node) > value)
                CachedKDTree_Traits<axis>::SetMax(node, value);
        }
        else
        {
            if(CachedKDTree_Traits<axis>::GetMax(node) > value)
            {
                CachedKDTree_Traits<axis>::SetMax(node, value);
                UpdateRecMax<axis>(node->child[0], value);
                UpdateRecMax<axis>(node->child[1], value);
            }
        }
    }

    template <int axis>
    void UpdateRecMin(NodeType * node, float value)
    {
        if(node->isLeaf)
        {
            if(CachedKDTree_Traits<axis>::GetMin(node) < value)
                CachedKDTree_Traits<axis>::SetMin(node, value);
        }
        else
        {
            if(CachedKDTree_Traits<axis>::GetMin(node) < value)
            {
                CachedKDTree_Traits<axis>::SetMin(node, value);
                UpdateRecMin<axis>(node->child[0], value);
                UpdateRecMin<axis>(node->child[1], value);
            }
        }
    }
    
    void WriteRec(NodeType * node, std::ostream & out) const
    {
        if(node->isLeaf)
        {
            out << node->element << std::endl;
        }
        else
        {
            WriteRec(node->child[0], out);
            WriteRec(node->child[1], out);
        }
    }

private:
    CachedKDTree_Node<Vec2Type> * root;

    Allocator m_poolAllocator;
};

}

#endif // CACHEDKDTREE_HPP
