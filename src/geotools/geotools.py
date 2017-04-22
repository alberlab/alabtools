# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_geotools', [dirname(__file__)])
        except ImportError:
            import _geotools
            return _geotools
        if fp is not None:
            try:
                _mod = imp.load_module('_geotools', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _geotools = swig_import_helper()
    del swig_import_helper
else:
    import _geotools
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _geotools.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _geotools.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _geotools.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _geotools.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _geotools.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _geotools.SwigPyIterator_equal(self, x)

    def copy(self):
        return _geotools.SwigPyIterator_copy(self)

    def next(self):
        return _geotools.SwigPyIterator_next(self)

    def __next__(self):
        return _geotools.SwigPyIterator___next__(self)

    def previous(self):
        return _geotools.SwigPyIterator_previous(self)

    def advance(self, n):
        return _geotools.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _geotools.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _geotools.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _geotools.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _geotools.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _geotools.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _geotools.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _geotools.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class DoubleVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, DoubleVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, DoubleVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _geotools.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _geotools.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _geotools.DoubleVector___bool__(self)

    def __len__(self):
        return _geotools.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _geotools.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _geotools.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _geotools.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _geotools.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _geotools.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _geotools.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _geotools.DoubleVector_pop(self)

    def append(self, x):
        return _geotools.DoubleVector_append(self, x)

    def empty(self):
        return _geotools.DoubleVector_empty(self)

    def size(self):
        return _geotools.DoubleVector_size(self)

    def swap(self, v):
        return _geotools.DoubleVector_swap(self, v)

    def begin(self):
        return _geotools.DoubleVector_begin(self)

    def end(self):
        return _geotools.DoubleVector_end(self)

    def rbegin(self):
        return _geotools.DoubleVector_rbegin(self)

    def rend(self):
        return _geotools.DoubleVector_rend(self)

    def clear(self):
        return _geotools.DoubleVector_clear(self)

    def get_allocator(self):
        return _geotools.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _geotools.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _geotools.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        this = _geotools.new_DoubleVector(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def push_back(self, x):
        return _geotools.DoubleVector_push_back(self, x)

    def front(self):
        return _geotools.DoubleVector_front(self)

    def back(self):
        return _geotools.DoubleVector_back(self)

    def assign(self, n, x):
        return _geotools.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _geotools.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _geotools.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _geotools.DoubleVector_reserve(self, n)

    def capacity(self):
        return _geotools.DoubleVector_capacity(self)
    __swig_destroy__ = _geotools.delete_DoubleVector
    __del__ = lambda self: None
DoubleVector_swigregister = _geotools.DoubleVector_swigregister
DoubleVector_swigregister(DoubleVector)


def ConvexHullIntersection(A, B):
    return _geotools.ConvexHullIntersection(A, B)
ConvexHullIntersection = _geotools.ConvexHullIntersection
# This file is compatible with both classic and new-style classes.

