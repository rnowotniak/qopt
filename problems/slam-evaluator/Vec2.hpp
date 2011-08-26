#ifndef VEC2_HPP
#define VEC2_HPP

template <typename T>
class Vec2
{
public:
    typedef T MyItemType;
    typedef T value_type;

    explicit Vec2()
    {}

    Vec2(T x, T y)
        : m_x(x), m_y(y)
    {}

    Vec2(const Vec2<T> & other)
    {
        m_x = other.m_x;
        m_y = other.m_y;
    }

    Vec2<T> & operator=(const Vec2<T> & other)
    {
        m_x = other.m_x;
        m_y = other.m_y;
        return *this;
    }

    inline Vec2<T> operator+(const Vec2<T> & other)
    {
        return Vec2<T>(m_x + other.m_x, m_y + other.m_y);
    }

    inline Vec2<T> & operator+=(const Vec2<T> & other)
    {
        m_x += other.m_x;
        m_y += other.m_y;
        return *this;
    }

    inline Vec2<T> Rotate(const T sinRotation, const T cosRotation)
    {
        return Vec2<T>(m_x * cosRotation - m_y * sinRotation, m_x * sinRotation + m_y * cosRotation);
    }

    inline T Distance2(const Vec2<T> & other) const
    {
        T x = m_x - other.m_x;
        T y = m_y - other.m_y;
        return x * x + y * y;
    }

    std::ostream & operator<<(std::ostream & out) const
    {
        out << m_x << " " << m_y;
        return out;
    }

    bool operator==(const Vec2<T> & other) const
    {
        return m_x == other.m_x && m_y == other.m_y;
    }

    bool operator!=(const Vec2<T> & other) const
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
};

template <typename T>
std::ostream & operator<<(std::ostream & out, const Vec2<T> & vec)
{
    vec << out;
    return out;
}

typedef Vec2<double> Vec2d;
typedef Vec2<float> Vec2f;

#endif //VEC2_HPP
