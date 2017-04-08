%module geotools
%{
#define SWIG_FILE_WITH_INIT
#include "geotools.h"
%}

%include "numpy.i"
%include "std_vector.i"

namespace std {
   %template(DoubleVector) vector<double>;
}
%init %{
import_array();
%}

%apply (double * IN_ARRAY1, int DIM1) {(double * A, int ASize), (double * B, int BSize)};

%include "geotools.h"
