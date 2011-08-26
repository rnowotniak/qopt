#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include <string>
#include <exception>

namespace Utils
{

class Exception: public std::exception
{
public:
    Exception(const std::string & what)
        : m_what(what)
    {}

    virtual ~Exception() throw() {}

    virtual const char * what() const throw() { return m_what.c_str(); }

private:
    const std::string m_what;
};

}

#endif // EXCEPTION_HPP
