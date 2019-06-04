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
%apply (float * IN_ARRAY1, int DIM1) {(float * Ax, int Ax_size), (float * radii, int radii_size), (float * values, int values_size)};
%apply (float * IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float * coordinates, int nbead, int nstruct, int dims)};
%apply (float * IN_ARRAY2, int DIM1, int DIM2) {(float * matrix, int row, int col), (float * coordinates, int row, int col)};
%apply (int * INPLACE_ARRAY1, int DIM1) {(int * Bi, int Bi_size),(int * Bj, int Bj_size)};
%apply (float * INPLACE_ARRAY1, int DIM1) {(float * Bx, int Bx_size)};
%apply (float * INPLACE_ARRAY2, int DIM1, int DIM2) {(float * confidence, int outi1, int outj1)};
%apply (float * INPLACE_ARRAY2, int DIM1, int DIM2) {(float * expected, int outi2, int outj2)};
%apply (float * INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float * Tomogram, int DimA, int DimB, int DimC)};
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

void BuildContactMap_func(float * coordinates, int nbead, int nstruct, int dims,
                          float * radii, int radii_size,
                          float contactRange,
                          int * Bi, int Bi_size,
                          int * Bj, int Bj_size,
                          float * Bx, int Bx_size)
{
    BuildContactMap(coordinates, nbead, nstruct, radii, contactRange, nbead, Bi, Bj, Bx);
}

void CalculatePixelConfidence(float * matrix, int row, int col,
                              float * confidence, int outi1, int outj1,
                              float * expected, int outi2, int outj2)
{
    PixelConfidence(matrix, row, col, confidence, expected);
}

void CalculateTomogramsFromStructure(float * coordinates, int row, int col,
                                     float * radii, int radii_size, float * values, int values_size,
                                     float radialExpansion, float sratio,
                                     float * Tomogram, int DimA, int DimB, int DimC)
{
    TomogramsFromStructure(coordinates, row,
                           radii, values, radialExpansion, sratio,
                           Tomogram, DimA, DimB, DimC);
}

%}
