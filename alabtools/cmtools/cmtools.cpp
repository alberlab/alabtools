#include<vector>
#include<algorithm>
#include<omp.h>
#include<iostream>
#include<cmath>
#include<limits>
#define THREADS 16
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
    std::vector<int> pBi[THREADS],pBj[THREADS];
    std::vector<float> pBx[THREADS];
    int top = 10;
    std::fill(Bi, Bi + DimB*(DimB+1)/2, 0);
    std::fill(Bj, Bj + DimB*(DimB+1)/2, 0);
    std::fill(Bx, Bx + DimB*(DimB+1)/2, 0);
#pragma omp parallel num_threads(THREADS)
{
    #pragma omp for schedule(dynamic, 5)
    for (int i = 0; i < DimB; ++i){
        if ((i+1) % int(DimB/10) == 0){
            std::cout << "=";
            std::cout.flush(); //process bar
        }

        int thread = omp_get_thread_num();
        for (int j = i; j < DimB; ++j){ //loop all indicies in new upper tril matrix
            int istart = mapping[i];
            int iend = mapping[i+1];
            int jstart = mapping[j];
            int jend = mapping[j+1];
            int dataSize = (iend - istart) * (jend - jstart);

            float *data = fetchData(Ap,Aj,Ax,istart,iend,jstart,jend,dataSize);

            std::sort(data,data + dataSize);

            float lowerFence,upperFence;
            boxplotStats(data,dataSize,lowerFence,upperFence);

            float topSum = 0;
            int topCount = 0;
            int topCut   = dataSize / top;

            if (topCut == 0){topCut = dataSize;}
            if (upperFence == 0){
                upperFence = 10;
                topCut     = dataSize;
            }

            for (int k = dataSize-1; k >= 0; --k){
                if (data[k] < upperFence){
                    topSum += data[k];
                    ++topCount;
                }
                if (topCount == topCut){break;}
            }
            delete[] data;
            pBi[thread].push_back(i);
            pBj[thread].push_back(j);
            pBx[thread].push_back(topSum / topCut);

        }
    }
}
    std::cout << std::endl;
    int k = 0;
    for (int i = 0; i < THREADS; ++i){
        for (int j = 0; j < pBi[i].size(); ++j){
            Bi[k] = pBi[i][j];
            Bj[k] = pBj[i][j];
            Bx[k] = pBx[i][j];
            ++k;
        }
    }
}

double SquareDistance(float x1, float y1, float z1, float x2, float y2, float z2){
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
}
void BuildContactMap(float * coordinates, //nbead*nstruct*3
                     int nbead,
                     int nstruct,
                     float * radii,
                     float contactRange,
                     int DimB,
                     int * Bi,
                     int * Bj,
                     float * Bx)
{
    std::vector<int> pBi[THREADS],pBj[THREADS];
    std::vector<float> pBx[THREADS];
    std::fill(Bi, Bi + DimB*(DimB+1)/2, 0);
    std::fill(Bj, Bj + DimB*(DimB+1)/2, 0);
    std::fill(Bx, Bx + DimB*(DimB+1)/2, 0);
    std::cout << nbead << " " << nstruct << std::endl;

#pragma omp parallel num_threads(THREADS)
{
    #pragma omp for schedule(dynamic, 5)
    for (int i = 0; i < DimB; ++i){
        if ((i+1) % int(DimB/10) == 0){
            std::cout << "=";
            std::cout.flush(); //process bar
        }
        int thread = omp_get_thread_num();
        for (int j = i; j < DimB; ++j){ //loop all indicies in new upper tril matrix
            double cap = (radii[i] + radii[j]) * contactRange;
            cap = cap*cap;
            float contacts = 0;
            for (int k = 0; k < nstruct; ++k){
                int indexI = i*nstruct*3 + k*3;
                int indexJ = j*nstruct*3 + k*3;
                double dist = SquareDistance(coordinates[indexI], coordinates[indexI+1], coordinates[indexI+2],
                                             coordinates[indexJ], coordinates[indexJ+1], coordinates[indexJ+2]);
                if (dist <= cap){
                    contacts = contacts + 1;
                }
            }
            pBi[thread].push_back(i);
            pBj[thread].push_back(j);
            pBx[thread].push_back(contacts / nstruct);

        }
    }
}
    std::cout << std::endl;
    int k = 0;
    for (int i = 0; i < THREADS; ++i){
        for (int j = 0; j < pBi[i].size(); ++j){
            Bi[k] = pBi[i][j];
            Bj[k] = pBj[i][j];
            Bx[k] = pBx[i][j];
            ++k;
        }
    }

}

const float Sxis[4][4] = {{ 0, 1, 3, 6},
                          {-1, 0, 2, 5},
                          {-3,-2, 0, 3},
                          {-6,-5,-3, 0}};
const float Sxi2s[4][4] = {{ 0, 1, 5,14},
                           { 1, 2, 6,15},
                           { 5, 6,10,19},
                           {14,15,19,28}};
const float Sxi3s[4][4] = {{  0,  1,  9, 36},
                           {  1,  0,  8, 35},
                           { -9, -8,  0, 27},
                           {-36,-35,-27,  0}};
const float Sxi4s[4][4] = {{  0,  1, 17, 98},
                           {  1,  2, 18, 99},
                           { 17, 18, 34,115},
                           { 98, 99,115,196}};

void predictQuadraticRegression(float * lhsValues, float * rhsValues, int lenl, int lenr, float & predictedValue, float & r){
    int n = lenl + lenr;
    if (n < 3){return;}

    float Sxi = Sxis[lenl][lenr];
    float Sxi2 = Sxi2s[lenl][lenr];
    float Sxi3 = Sxi3s[lenl][lenr];
    float Sxi4 = Sxi4s[lenl][lenr];

    float Syi = 0, Sxiyi = 0, Sxi2yi = 0;

    for (int i = 0; i<lenl; ++i){
        Syi += lhsValues[i];
        Sxiyi += -(i+1)*lhsValues[i];
        Sxi2yi += (i+1)*(i+1)*lhsValues[i];
    }
    for (int i = 0; i<lenr; ++i){
        Syi += rhsValues[i];
        Sxiyi += (i+1)*rhsValues[i];
        Sxi2yi += (i+1)*(i+1)*rhsValues[i];
    }

    float meanx = Sxi/n, meany = Syi/n, meanx2 = Sxi2/n;
    float Sxx = Sxi2/n - meanx*meanx;
    float Sxy = Sxiyi/n - meanx*meany;
    float Sxx2 = Sxi3/n - meanx*meanx2;
    float Sx2x2 = Sxi4/n - meanx2*meanx2;
    float Sx2y = Sxi2yi/n - meanx2*meany;

    float coefDenomin = Sxx*Sx2x2 - Sxx2*Sxx2;
    if (coefDenomin == 0){return;}

    float B = (Sxy*Sx2x2 - Sx2y*Sxx2)/coefDenomin;
    float A = (Sx2y*Sxx - Sxy*Sxx2)/coefDenomin;
    float C = (Syi - B*Sxi - A*Sxi2)/n;

    if (C <= 0){return;}

    float SSE = 0;
    float SST = 0;

    float error, total;
    for (int i = 0; i<lenl; ++i){
        error = lhsValues[i] - A*(i+1)*(i+1) + B*(i+1) - C;
        total = lhsValues[i] - meany;

        SSE += error*error;
        SST += total*total;
    }
    for (int i = 0; i<lenr; ++i){
        error = rhsValues[i] - A*(i+1)*(i+1) - B*(i+1) - C;
        total = rhsValues[i] - meany;

        SSE += error*error;
        SST += total*total;
    }

    if (SST == 0){return;}

    float rnow = std::sqrt(1-SSE/SST);

    if (rnow > r){
        predictedValue = C;
        r = rnow;
    }
}

void predictExponentialRegression(float * lhsValues, float * rhsValues, int lenl, int lenr, float & predictedValue, float & r){
    int n = lenl + lenr;
    if (n < 3){return;}

    float Sxi = Sxis[lenl][lenr];
    float Sxi2 = Sxi2s[lenl][lenr];
    float Slnyi = 0, Sxilnyi = 0, Slnyi2 = 0;

    for (int i = 0; i<lenl; ++i){
        if (lhsValues[i] <= 0) {return;}
        Slnyi   += std::log(lhsValues[i]);
        Sxilnyi += -(i+1)*std::log(lhsValues[i]);
        Slnyi2  += std::pow(std::log(lhsValues[i]),2);
    }
    for (int i = 0; i<lenl; ++i){
        if (rhsValues[i] <= 0) {return;}
        Slnyi   += std::log(rhsValues[i]);
        Sxilnyi += (i+1)*std::log(rhsValues[i]);
        Slnyi2  += std::pow(std::log(rhsValues[i]),2);
    }

    float meanx = Sxi/n, meanlny = Slnyi/n;

    float Sxx = Sxi2/n - meanx * meanx;
    float Syy = Slnyi2/n - meanlny * meanlny;
    float Sxy = Sxilnyi/n - meanx * meanlny;

    if ((Sxx == 0) or (Syy == 0)){return;}

    float B = std::exp(Sxy/Sxx);
    float A = std::exp(meanlny - meanx * std::log(B));

    float rnow = std::abs(Sxy)/std::sqrt(Sxx*Syy);

    if (rnow > r){
        predictedValue = A;
        r = rnow;
    }
}

const int iincRHS[4][3] = {{1, 2, 3},
                           {1, 2, 3},
                           {0, 0, 0},
                           {-1, -2, -3}};
const int jincRHS[4][3] = {{0, 0, 0},
                           {1, 2, 3},
                           {1, 2, 3},
                           {1, 2, 3}};
void PixelConfidence(float * matrix, int row, int col, //matrix and size
                     float * confidence,
                     float * expected)
{
#pragma omp parallel num_threads(THREADS)
{
    #pragma omp for schedule(dynamic, 5)
    for (int i = 0; i < row; ++i){
        //std::cout << i << std::endl;
        for (int j = 0; j < col; ++j){
            float currentValue = matrix[i*col + j];
            float valsum = std::numeric_limits<float>::max(), valweight = 0, expval = 0, expweight = 0;
            for (int k = 0; k < 4; ++k){
                float lhsValues[3] = {0,0,0};
                float rhsValues[3] = {0,0,0};
                int lenl = 0, lenr = 0;

                for (int l = 0; l < 3; ++l){
                    int irhs = i + iincRHS[k][l];
                    int ilhs = i - iincRHS[k][l];

                    int jrhs = j + jincRHS[k][l];
                    int jlhs = j - jincRHS[k][l];

                    if ((ilhs >= 0) and (ilhs < row) and
                        (jlhs >= 0) and (jlhs < col)){
                        lhsValues[l] = matrix[ilhs*col + jlhs];
                        lenl++;
                    }//0 otherwise

                    if ((irhs >= 0) and (irhs < row) and
                        (jrhs >= 0) and (jrhs < col)){
                        rhsValues[l] = matrix[irhs*col + jrhs];
                        lenr++;
                    }//0 otherwise
                }

                float predictedValue = 0, r = 0;

                predictQuadraticRegression(lhsValues, rhsValues, lenl, lenr, predictedValue, r);
                predictExponentialRegression(lhsValues, rhsValues, lenl, lenr, predictedValue, r);
                if ( !std::isnan(predictedValue) and (predictedValue != 0)){
                    valsum = valsum * valweight + std::abs(currentValue - predictedValue)/predictedValue * r * ((lenl+lenr)/6);
                    valweight += 1; //this is one because we are taking average of 4 predictions
                    valsum /= valweight;
                    
                    expval = expval * expweight + predictedValue * r * ((lenl+lenr)/6);
                    expweight += r * ((lenl+lenr)/6);
                    expval /= expweight;
                }
                
                

            }//k
            
            confidence[i*col+j] = std::exp(-valsum);
            if (std::isnan(expval)){ expval = 0; }
            expected[i*col+j] = expval;
            //printf("%d %d %f %f\n",i, j, currentValue, confidence[i*col+j]);
        }
    }
}

}
/*
int main(){
    std::ifstream f;
    f.open("Ap.txt");
    int Ap[21];
    for (int i = 0; i< 21; i++){
        f >> Ap[i];
        //std::cout << Ap[i] << ' ';
    }
    f.close();
    f.open("Aj.txt");
    int Aj[399];
    for (int i = 0; i< 399; i++){
        f >> Aj[i];
        //std::cout << Aj[i] << ' ';
    }
    f.close();
    f.open("Ax.txt");
    float Ax[399];
    for (int i = 0; i< 399; i++){
        f >> Ax[i];
        //std::cout << Ax[i] << ' ';
    }
    f.close();

    int mapping[3] = {0,10,20};
    int *Bi = new int[3];
    int *Bj = new int[3];
    float *Bx = new float[3];
    TopmeanSummaryMatrix(Ap,Aj,Ax,20,2,mapping,Bi,Bj,Bx);

    for (int i = 0; i < 3; i++){
        std::cout << Bi[i] << Bj[i] << Bx[i] << std::endl;
    }
}
*/
