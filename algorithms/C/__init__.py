import ctypes
import os, os.path


class AlgorithmInC:
    def __init__(self):
        filename = str(self.__class__).split('.')[-1]
        lib = ctypes.CDLL(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.so' % filename))
        self.__dict__['lib'] = lib

    def __getattr__(self, name):
        print name
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        if self.__class__.interface.has_key(name):
            return self.__class__.interface[name].in_dll(self.__dict__['lib'], name)
        return getattr(self.lib, name)

    def __setattr__(self, name, value):
        if self.__class__.interface.has_key(name):
            if self.__class__.interface[name] == ctypes.c_void_p:
                ctypes.c_void_p.in_dll(self.__dict__['lib'], name).value = \
                        ctypes.cast(value, ctypes.c_void_p).value
            else:
                self.__class__.interface[name].in_dll(self.__dict__['lib'], name).value = value
        else:
            raise AttributeError(name)

class parfoobar(ctypes.Structure):
    _fields_ = [('param1', ctypes.c_int)]

class rQIEA(AlgorithmInC):
    def __init__(self):
        AlgorithmInC.__init__(self)

    class Ind(ctypes.Structure):
        _fields_ = [('genotype', ctypes.c_void_p), ('fitness', ctypes.c_longdouble)]


    class foobar(ctypes.Structure):
        _fields_ = [
                ('parent', ctypes.POINTER(parfoobar)),
                ('somepar', ctypes.c_int),
                ('param1', ctypes.c_int),
                ]

    interface = {
            'Pc'        : ctypes.c_float,
            'popsize'   : ctypes.c_int,
            'tmax'      : ctypes.c_int,
            'evaluator' : ctypes.c_void_p,
            'best'      : ctypes.POINTER(Ind),
            }

