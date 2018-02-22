void TopmeanSummaryMatrix(int * Ap,
                          int * Aj,
                          float * Ax,
                          int DimA,
                          int DimB,
                          int * mapping,
                          int * Bi,
                          int * Bj,
                          float * Bx);
void BuildContactMap(float * coordinates,
                     int nbead,
                     int nstruct,
                     float * radii,
                     float contactRange,
                     int DimB,
                     int * Bi,
                     int * Bj,
                     float * Bx);
void PixelConfidence(float * matrix, int n, //matrix and size
                     float * confidence);
