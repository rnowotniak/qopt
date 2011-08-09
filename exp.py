#!/usr/bin/python
#
# Glowny uruchamiacz pojedynczego eksperymentu obliczeniego
# Odpowiedzialnosc:
#     parsowanie argumentow, importowanie klas, ustawienia parametrow, uruchomienie algorytmu,
#     ewentualnie takze np. interaktywny shell, generowanie pliku z logiem.
#

import getopt,sys,os,time,random
import framework
import numpy
import subprocess

sys.path.insert(0, os.getcwd() + '/problems')

# set attributes of the object according to the args array elements
def setAttrs(obj, args):
    opts = []
    i = 0
    for i in xrange(0, len(args), 2):
        a = args[i]
        if a.startswith('-'):
            a = a.strip('-')
            setattr(obj, a, type(getattr(obj,a,''))(args[i+1]))
        else:
            return args[i:]
    return args[i+2:]

# creates a wrapper for any callable obj object
# when the wrapped object is called, internal evaluation counter of the object is incremented
def CounterWrapper(obj):
    origcall = obj.__call__
    def wrap(*args, **kwargs):
        obj.evaluationCounter += 1
        return origcall(*args, **kwargs)
    obj.__call__ = wrap
    obj.evaluationCounter = 0
    return obj

class Experiment():
    def __init__(self, argv):
        # parse the main options provided
        (options,args) = getopt.getopt(argv[1:],
                "l:p:s:o:",
                ('logfile=', 'postprocess=', 'seed='))
        print 'options: ' + str(options)

        self.logfile = '/tmp/log.txt'
        framework.PRNGseed = time.time()
        for (o,a) in options:
            if o in ('-l', '--logfile'):
                self.logfile = a
            elif o in ('-p', '--postprocess'):
                # postprocess the given logfile
                framework.Logging.postprocess(a)
                sys.exit(0)
            elif o in ('-s', '--seed'):
                framework.PRNGseed = float(a)

        # initialize the random numbers generator with the seed
        random.seed(framework.PRNGseed)

        # formatting options for numpy
        numpy.set_printoptions(linewidth=float('inf'))

        # load the algorithm
        algo = args[0]
        print 'algorithm:' + algo
        try:
            # try to load the algorithm as a Python class
            self.algoobj = None
            name = algo.rsplit('.',1)
            Class = getattr(__import__(name[0]), name[1]) # import module.Class
            self.algoobj = Class() # create the algorithm instance
        except Exception, e:
            print e
        if self.algoobj:
            # Python class for the algorithm loaded successfuly
            if os.path.exists(algo):
                raise Exception('Both Python class and executable %s exist. Aborting' % algo)
            # parse the algorithm options
            args = setAttrs(self.algoobj, args[1:])
            print 'evaluator: ' + str(args)
            # load the evaluator
            try:
                name = args[0].rsplit('.',1)
                Class = getattr(__import__(name[0]), name[1])
                self.algoobj.evaluator = Class() # create the evaluator instance
                self.algoobj.evaluator = CounterWrapper(self.algoobj.evaluator) # wrap with counter
            except Exception, e:
                print e
                raise Exception("Failed to load the evaluator class %s" % args[0])
            args = setAttrs(self.algoobj.evaluator, args[1:])
        elif os.path.exists(algo):
            # TODO: run the algorithm as a binary executable (create the logfile and redirect the process output there)
            self.algoobj = framework.ExecutableAlgorithm(*args)
            #print args
            #subprocess.Popen(args)
        else:
            raise Exception('Neither class nor executable file %s exist. Aborting' % algo)

if __name__ == '__main__':
    e = Experiment(sys.argv)

    # involve the logging subsystem and run the algorithm
    logging = framework.Logging(e.logfile, e.algoobj)
    e.algoobj.run() # run the algorithm
    logging.close()

## Interactive shell
#
# import code, traceback, signal
# 
# def debug(sig, frame):
#     """Interrupt running process, and provide a python prompt for
#     interactive debugging."""
#     d={'_frame':frame}         # Allow access to frame object.
#     d.update(frame.f_globals)  # Unless shadowed by global
#     d.update(frame.f_locals)
# 
#     i = code.InteractiveConsole(d)
#     message  = "Signal recieved : entering python shell.\nTraceback:\n"
#     message += ''.join(traceback.format_stack(frame))
#     i.interact(message)
# 
# def listen():
#     signal.signal(signal.SIGUSR1, debug)  # Register handler

