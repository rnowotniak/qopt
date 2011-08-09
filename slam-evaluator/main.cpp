#include <fstream>
#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "Transformation.hpp"

#include "CachedKDTree.hpp"
#include "CachedKDTree_Vec2.hpp"
#include "CachedKDTree_Vector.hpp"
#include "CachedKDTree_TransformScan.hpp"

template <typename HelpObject>
void PrintHelpAndExit(const HelpObject & help)
{
    std::cerr << help << std::endl;
    exit(0);
}

template <typename OutIterator>
OutIterator ReadMapFromFile(const std::string & file, OutIterator out)
{
    std::ifstream input(file.c_str());

    if(!input)
        throw std::invalid_argument(file + " unavaliable");

    //std::cerr << "reading map" << std::endl;

    while(1)
    {
        float x, y;
        char c; // for: ',';
        input >> x;
        input >> c;
        input >> y;
        if(input.eof() == true)
            break;
        *out++ = Utils::CachedKDTree_Vec2<float>(x, y);
    }
    
    //std::cerr << "reading map done" << std::endl;
    return out;
}

template <typename OutIterator>
OutIterator ReadScanFromFile(const std::string & file, OutIterator out)
{
    std::ifstream input(file.c_str());

    if(!input)
        throw std::invalid_argument(file + " unavaliable");

    float dx, dy, da;
    int count;
    char c; // for ','

    input >> dx >> c >> dy >> c >> da >> count;
    
    //std::cerr << "reading scan" << std::endl;

    while(1)
    {
        float x, y;
        input >> x;
        input >> c;
        input >> y;
        if(input.eof() == true)
            break;
        *out++ = Utils::CachedKDTree_Vec2<float>(x, y);
    }

    //std::cerr << "reading scan done" << std::endl;
    return out;
}

void Run(const std::string & mapFile, const std::string & scanFile, bool constTreshold, int executionsInSingleIteration)
{
    typedef Utils::CachedKDTree_Vector<Utils::CachedKDTree_Vec2<float> > CachedVector;

    std::vector<Utils::CachedKDTree_Vec2<float> > kdTreePoints;
    std::vector<Utils::CachedKDTree_Vec2<float> > scanPoints;

    kdTreePoints.reserve(10000);
    scanPoints.reserve(1081);

    ReadMapFromFile(mapFile, std::back_inserter(kdTreePoints));
    ReadScanFromFile(scanFile, std::back_inserter(scanPoints));

    //std::cerr << "kdtree begin" << std::endl;
    Utils::CachedKDTree<Utils::CachedKDTree_Vec2<float> > kdTree(kdTreePoints.begin(), kdTreePoints.end());
    //std::cerr << "kdtree end" << std::endl;

    int errorIterations = executionsInSingleIteration;
    int iterations = 0;

    float errorTreshold = std::numeric_limits<float>::max();

    while(1)
    {
        float dx, dy, da;

        //std::cerr << "waiting" << std::endl;

        std::cin >> dx >> dy >> da;

        if(std::cin.good() == false)
            break;

        Transformation t(dx, dy, da);

        CachedVector points;
        points.resize(scanPoints.size());
        
        Utils::CachedKDTree_TransformScan transformation(t);
        transformation.Transform(scanPoints.begin(), scanPoints.end(), points.begin());

        float Itk = 0.0f;
        int ntk = 0;

        for(CachedVector::iterator it = points.begin(); it != points.end(); ++it)
        {
            Utils::CachedKDTree_Vec2<float> nn;
            float distance;

            kdTree.FindNN(*it, nn, distance);

            Itk += distance;
            
            if(distance <= errorTreshold)
            {
                ++ntk;
            }
        }

        --errorIterations;

        if(errorIterations == 0)
        {
            errorIterations = executionsInSingleIteration;
            if(!constTreshold)
            {
                if(iterations == 7)
                {
                    errorTreshold = 3.0f;
                }
                else if(iterations > 7)
                {
                    errorTreshold = errorTreshold * 0.6f + 0.01f * 0.4f; // lim->1cm
                }
            }
        }

        if(ntk > 0)
        {
            std::cout << (Itk * points.size()) / ntk << std::endl;
        }
        else
        {
            std::cout << std::numeric_limits<float>::max() << std::endl;
        }
        iterations++;
    }
}

int main(int ac, char* av[])
{
    namespace po = boost::program_options;

    try
    {
        po::options_description desc("Allowed options");

        desc.add_options()
            ("help", "print this help")
            ("map-file", po::value<std::string>(), "file containing map points")
            ("scan-file", po::value<std::string>(), "file containing single scan file")
            ("const-treshold", "error treshold will not decrease during iterations.")
            ("executions-in-single-iteration", po::value<int>(), "error treshold will desrease after _arg_ iterations. [default 1]");

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            PrintHelpAndExit(desc);
        }

        if(!vm.count("map-file"))
        {
            std::cout << "Map file not set" << std::endl;
            PrintHelpAndExit(desc);
        }

        if(!vm.count("scan-file"))
        {
            std::cout << "Scan file not set" << std::endl;
            PrintHelpAndExit(desc);
        }

        std::string mapFile = vm["map-file"].as<std::string>();
        std::string scanFile = vm["scan-file"].as<std::string>();

        bool constTreshold = false;
        if(vm.count("const-treshold"))
            constTreshold = true;

        int executionsInSingleIteration = 1;
        if(vm.count("executions-in-single-iteration"))
            executionsInSingleIteration = vm["executions-in-single-iteration"].as<int>();

        if(executionsInSingleIteration <= 0)
        {
            std::cout << "executions-in-single-iteration should be grather than 0" << std::endl;
            PrintHelpAndExit(desc);
        }

        Run(mapFile, scanFile, constTreshold, executionsInSingleIteration);

    }
    catch(std::exception & e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "Unknown exception" << std::endl;
    }

#if defined(WINDOWS)
    std::system("PAUSE");
#endif

    return 0;
}
