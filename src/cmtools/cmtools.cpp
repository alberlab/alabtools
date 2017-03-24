#include<vector>
#include<algorithm>
#include<omp.h>
#include<iostream>
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
    float *data = new float[spaceLength];
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
    int dataSize = 10*10;
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
            std::cout.flush();
        }
        
        int thread = omp_get_thread_num();
        for (int j = i; j < DimB; ++j){ //loop all indicies in new upper tril matrix
            int istart = mapping[i];
            int iend = mapping[i+1];
            int jstart = mapping[j];
            int jend = mapping[j+1];
            
            float *data = fetchData(Ap,Aj,Ax,istart,iend,jstart,jend,dataSize);
            
            std::sort(data,data+dataSize);
            
            float lowerFence,upperFence;
            boxplotStats(data,dataSize,lowerFence,upperFence);

            float topSum = 0;
            int topCount = 0;
            int topCut   = top;
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
            pBx[thread].push_back(topSum/topCut);
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
