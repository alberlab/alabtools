%module cmtools
%{
#define SWIG_FILE_WITH_INIT
#include "cmtools.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%include "cmtools.h"

%apply (int * IN_ARRAY1, int DIM1) {(int * Ap, int AP_size), (int * Aj, int Aj_size), (int * mapping, int mapping_size)};
%apply (float * IN_ARRAY1, int DIM1) {(float * Ax, int Ax_size)};
%apply (int * INPLACE_ARRAY1, int DIM1) {(int * Bi, int Bi_size),(int * Bj, int Bj_size)};
%apply (float * INPLACE_ARRAY1, int DIM1) {(float * Bx, int Bx_size)};
%inline %{
void TopmeanSummaryMatrix_func(int * Ap, int AP_size,
                               int * Aj, int Aj_size,
                               float * Ax, int Ax_size,
                               int DimA,
                               int DimB,
                               int * mapping, int mapping_size,
                               int * Bi, int Bi_size,
                               int * Bj, int Bj_size,
                               float * Bx, int Bx_size)
{
    TopmeanSummaryMatrix(Ap, Aj, Ax, DimA, DimB, mapping, Bi, Bj, Bx);
}

%}
