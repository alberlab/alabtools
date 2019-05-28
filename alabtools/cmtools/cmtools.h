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
void PixelConfidence(float * matrix, int row, int col, //matrix and size
                     float * confidence,
                     float * expected);

void TomogramsFromStructure(float * coordinates, int nbead,
                            float * radii, float radialExpansion, float sratio,
                            float * Tomogram, int DimA, int DimB, int DimC);
