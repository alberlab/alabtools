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

%include "geotools.h"

%apply (double * IN_ARRAY1, int DIM1) {(double * A, int ASize), (double * B, int BSize)};

%apply (float * IN_ARRAY1, int DIM1) {(float * radii, int radii_size)};

%apply (float * IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float * coordinates, int nbead, int nstruct, int dims)};

%apply (float * INPLACE_ARRAY2, int DIM1, int DIM2) {(float * results, int outdim1, int outdim2)};


%inline %{
void BoundingSpheresWrapper(float * coordinates, int nbead, int nstruct, int dims,
                            float * radii, int radii_size,
                            float * results, int outdim1, int outdim2)
{
    bounding_spheres(coordinates, radii, nbead, nstruct, results);
}
%}
