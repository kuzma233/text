#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <windows.h>
#include <immintrin.h> //AVX、AVX2
using namespace std;


// 初始化矩阵
void ini_Matrix(float** &A, float* &b, int n) {
    srand(time(NULL));
    
    // 动态分配矩阵 A
    A = new float*[n];
    for (int i = 0; i < n; ++i) {
        A[i] = new float[n];
        for (int j = 0; j < n; ++j) {
            A[i][j] = float(rand() % 10 + 1); // 生成1到10的随机数填充矩阵
        }
    }
    
    // 动态分配向量 b
    b = new float[n];
    for (int i = 0; i < n; ++i) {
        b[i] = float(rand() % 10 + 1); // 生成1到10的随机数填充向量b
    }
}

void ini_Matrix_align(float** &A, float* &b, int n,int alignment) {
    srand(time(NULL));
    
    // 动态分配矩阵 A
    A = (float**)_aligned_malloc(sizeof(float*) * n, alignment);
    for (int i = 0; i < n; i++) {
        A[i] = (float*)_aligned_malloc(sizeof(float) * n, alignment);
        //使得矩阵每一行在内存中按照alignment对齐，SSE为16，AVX为32
        for (int j = 0; j < n; ++j) {
            A[i][j] = float(rand() % 10 + 1); // 生成1到10的随机数填充矩阵
        }
    }
    
    // 动态分配向量 b
    b = (float*)_aligned_malloc(sizeof(float) * n, alignment);
    for (int i = 0; i < n; ++i) {
        b[i] = float(rand() % 10 + 1); // 生成1到10的随机数填充向量b
    }
}


// 高斯消去
void gaussian(float **A, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        // 高斯消元步骤
        for (int j = i + 1; j < n; ++j) {
            float factor = A[j][i] / A[i][i];
            for (int k = i; k < n; ++k) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
}

void gaussian_AVX(float** A, float* b, int n) {
    for (int k = 0; k < n; k++) {
        __m256 t1 = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 t2 = _mm256_loadu_ps(&A[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&A[k][j], t2);
        }
        for (; j < n; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void gaussian_AVX_align(float** A, float* b, int n)
{
    for (int k = 0; k < n; k++) {
        int j = k+1;
        while (j%8)
        {
            A[k][j] = A[k][j] / A[k][k];
            j++;
        }
        __m256 t1 = _mm256_set1_ps(A[k][k]);
        for ( ; j + 8 <= n; j += 8) {
            __m256 t2 = _mm256_load_ps(&A[k][j]);   //已对齐，用load和store指令
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(&A[k][j], t2);
        }
        for (; j < n; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        b[k] = b[k]/A[k][k];
        for (int i = k + 1; i < n; i++) {
            j = k + 1;
            while (j%8)
            {
                A[i][j] -= A[i][k] * A[k][j];
                j++;
            }
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for ( ; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_load_ps(&A[k][j]);
                __m256 vij = _mm256_load_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_store_ps(&A[i][j], vij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
            b[i] = b[i] - A[i][k]*b[k];
        }
    }
}


// 回代
void backup(float** A, float* b, float x[], int n) {
    x[n-1] = b[n-1] / A[n-1][n-1];
    float sum;
    for (int i = n - 1; i >= 0; --i) {
        sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j]* x[j];
        }
        x[i] = sum / A[i][i];
    }
}

void backup_AVX(float** A, float* b, float x[], int n) {
    x[n-1] = b[n-1] / A[n-1][n-1];
    float sum;
    float* sum_result;
    for (int i = n - 1; i >= 0; --i) {
        sum = b[i];
        int j=0;
        for(j = i+1; j+8 < n; j+=8){
            __m256 vaij = _mm256_loadu_ps(&A[i][j]);
            __m256 vax = _mm256_loadu_ps(&x[j]);
            __m256 vx = _mm256_mul_ps(vaij, vax);
            __m256 temp = _mm256_hadd_ps(vx, vx);
            __m256 result0 = _mm256_hadd_ps(temp, temp);
            __m256 result = _mm256_hadd_ps(result0, result0);
            _mm256_storeu_ps(sum_result, result);
            sum -= sum_result[0];
        }
        for( ; j < n; j++){
            sum -= A[i][j]* x[j];
        }
        x[i] = sum / A[i][i];
    }
}


//主进程计时
void Gaussian_Elimination(int n){
    float** A;
    float* b;
    float* x;
    x=new float[n];
    ini_Matrix(A,b,n);
    
    long long freq,start,end;
    QueryPerformanceFrequency((LARGE_INTEGER *) & freq);

    QueryPerformanceCounter((LARGE_INTEGER*)&start);
    gaussian(A, b, n);
    backup(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end);

    double interval = (end - start) * 1000.0 / freq;
    cout<<"-----------普通算法-------------"<<endl;
    cout <<"Total time:"<<interval <<"ms" <<endl;
}

void Gaussian_Elimination_AVX(int n){
    float** A;
    float* b;
    float* x;
    x=new float[n];
    ini_Matrix(A,b,n);

    long long freq,start,end;
    QueryPerformanceFrequency((LARGE_INTEGER *) & freq);

    QueryPerformanceCounter((LARGE_INTEGER*)&start);
    gaussian_AVX(A, b, n);
    backup_AVX(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end);

    double interval = (end - start) * 1000.0 / freq;
    cout<<"---------AVX并行优化算法-----------"<<endl;
    cout <<"Total time:"<<interval <<"ms" <<endl;
}

void Gaussian_Elimination_AVX_align(int n){
    float** A;
    float* b;
    float* x;
    x=new float[n];
    ini_Matrix_align(A,b,n,32);

    long long freq,start,end;
    QueryPerformanceFrequency((LARGE_INTEGER *) & freq);

    QueryPerformanceCounter((LARGE_INTEGER*)&start);
    gaussian_AVX_align(A, b, n);
    backup_AVX(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end);

    double interval = (end - start) * 1000.0 / freq;
    cout<<"-------AVX并行优化算法（对齐）---------"<<endl;
    cout <<"Total time:"<<interval<<"ms" <<endl;
}


int main() {
    cout<<"问题规模：500"<<endl;
    Gaussian_Elimination(500);
    Gaussian_Elimination_AVX(500);
    Gaussian_Elimination_AVX_align(500);
    cout<<endl;
    
    cout<<"问题规模：1000"<<endl;
    Gaussian_Elimination(1000);
    Gaussian_Elimination_AVX(1000);
    Gaussian_Elimination_AVX_align(1000);
    cout<<endl;

    cout<<"问题规模：2000"<<endl;
    Gaussian_Elimination(2000);
    Gaussian_Elimination_AVX(2000);
    Gaussian_Elimination_AVX_align(2000);
    cout<<endl;

    cout<<"问题规模：3000"<<endl;
    Gaussian_Elimination(3000);
    Gaussian_Elimination_AVX(3000);
    Gaussian_Elimination_AVX_align(3000);
    cout<<endl;

    cout<<"问题规模：4000"<<endl;
    Gaussian_Elimination(4000);
    Gaussian_Elimination_AVX(4000);
    Gaussian_Elimination_AVX_align(4000);
    cout<<endl;

}
