#include<vector>
#include<algorithm>
#include<iostream>
#include<cmath>
#include<limits>
#include<cstdio>

float *fetchData(int * Ap,
                 int * Aj,
                 float * Ax,
                 int istart,
                 int iend,
                 int jstart,
                 int jend,
                 int spaceLength)
{
    //prepare newdata
    float *data = new float[spaceLength+1];
    std::fill(data,data+spaceLength,0);
    //pointer to location for insert
    int pd = 0;

    for (int i = istart; i < iend; ++i){
        int *low;
        int *up;

        low = std::lower_bound(Aj+Ap[i],Aj+Ap[i+1],jstart);
        up  = std::upper_bound(Aj+Ap[i],Aj+Ap[i+1],jend-1);

        int dlow = low -Aj;
        int dup  = up - Aj;
        for (int j = dlow; j != dup; ++j){
            data[pd] = Ax[j];
            if ((istart==jstart) && (iend==jend)){//duplicate item for diagonal
                if (Aj[j] > i){
                    data[++pd] = Ax[j];
                }else{
                    pd--;
                }
            }
            ++pd;
        }
    }
    return data;
}

void boxplotStats(const float * data,
                  const int dataSize,
                  float &lowerFence,
                  float &upperFence)
{
    float Q1 = data[int(dataSize*0.25+0.5)];
    float Q3 = data[int(dataSize*0.75+0.5)];
    upperFence = Q3 + 1.5*(Q3-Q1);
    lowerFence = Q1 - 1.5*(Q3-Q1);
}

/*
 * Ap, Aj, Ax matrix ptr, indicies, data
 * DimA : A dimension
 * mapping : length DimB+1; B's summary matrix range. ith row correspones to mapping[i], mapping[i+1]-1
 * Bi,Bj,Bx has DimB*(DimB+1)/2
 */
void TopmeanSummaryMatrix(int * Ap,
                          int * Aj,
                          float * Ax,
                          int DimA,
                          int DimB,
                          int * mapping,
                          int * Bi,
                          int * Bj,
                          float * Bx)
{
    int top = 10;

    // initialize output
    std::fill(Bi, Bi + DimB*(DimB+1)/2, 0);
    std::fill(Bj, Bj + DimB*(DimB+1)/2, 0);
    std::fill(Bx, Bx + DimB*(DimB+1)/2, 0);

    int out_k = 0;

    for (int i = 0; i < DimB; ++i){
        if ((i+1) % int(DimB/10) == 0){
            std::cout << "=";
            std::cout.flush(); // progress bar
        }

        for (int j = i; j < DimB; ++j){ // upper triangular
            int istart = mapping[i];
            int iend   = mapping[i+1];
            int jstart = mapping[j];
            int jend   = mapping[j+1];
            int dataSize = (iend - istart) * (jend - jstart);

            float *data = fetchData(Ap, Aj, Ax,
                                     istart, iend,
                                     jstart, jend,
                                     dataSize);

            std::sort(data, data + dataSize);

            float lowerFence, upperFence;
            boxplotStats(data, dataSize, lowerFence, upperFence);

            float topSum = 0.0f;
            int topCount = 0;
            int topCut   = dataSize / top;

            if (topCut == 0) {
                topCut = dataSize;
            }
            if (upperFence == 0) {
                upperFence = 10;
                topCut     = dataSize;
            }

            for (int k = dataSize - 1; k >= 0; --k){
                if (data[k] < upperFence){
                    topSum += data[k];
                    ++topCount;
                }
                if (topCount == topCut){
                    break;
                }
            }

            delete[] data;

            Bi[out_k] = i;
            Bj[out_k] = j;
            Bx[out_k] = topSum / topCut;
            ++out_k;
        }
    }

    std::cout << std::endl;
}
