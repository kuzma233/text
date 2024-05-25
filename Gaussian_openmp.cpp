#include<iostream>
#include <ctime>  //随机种子
#include <windows.h> //计时
#include <immintrin.h> //AVX、AVX2
#include <omp.h>  //openmp
#define NUM_THREADS 7
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


//avx & openmp
void avx_omp_static(float** A,float* b,int N) {            //avx static
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);   
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            b[k] = b[k]/A[k][k];
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k]*b[k];
            A[i][k] = 0;
        }
    }
}

void avx_omp_dynamic(float** A,float* b,int N) {            //avx dynamic
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);   
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            b[k] = b[k]/A[k][k];
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k]*b[k];
            A[i][k] = 0;
        }
    }
}

void avx_omp_guided(float** A,float* b,int N) {            //avx guided
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]); 
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            b[k] = b[k]/A[k][k];
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k]*b[k];
            A[i][k] = 0;
        }
    }
}


//主进程计时
void Gaussian_Elimination(int n){
    float** A;
    float* b;
    float* x;
    x=new float[n];
    
    long long freq,start_0,end_0,start_1,end_1,start_2,end_2,start_3,end_3;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    ini_Matrix(A,b,n);
    QueryPerformanceCounter((LARGE_INTEGER*)&start_0);
    gaussian(A, b, n);
    backup(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end_0);

    ini_Matrix(A,b,n);
    QueryPerformanceCounter((LARGE_INTEGER*)&start_1);
    avx_omp_static(A, b, n);
    backup(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end_1);

    ini_Matrix(A,b,n);
    QueryPerformanceCounter((LARGE_INTEGER*)&start_2);
    avx_omp_dynamic(A, b, n);
    backup(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end_2);

    ini_Matrix(A,b,n);
    QueryPerformanceCounter((LARGE_INTEGER*)&start_3);
    avx_omp_guided(A, b, n);
    backup(A, b, x, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end_3);

    double interval_0 = (end_0 - start_0) * 1000.0 / freq;
    cout <<"串行算法 :"<<interval_0 <<"ms" <<endl;
    double interval_1 = (end_1 - start_1) * 1000.0 / freq;
    cout <<"static :"<<interval_1 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_1<<endl;
    double interval_2 = (end_2 - start_2) * 1000.0 / freq;
    cout <<"dynamic :"<<interval_2 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_2<<endl;
    double interval_3 = (end_3 - start_3) * 1000.0 / freq;
    cout <<"guided :"<<interval_3 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_3<<endl;
}


int main() {
    cout<<"问题规模：1000"<<endl;
    Gaussian_Elimination(1000);
    cout<<endl;

    cout<<"问题规模：1500"<<endl;
    Gaussian_Elimination(1500);
    cout<<endl;

    cout<<"问题规模：2000"<<endl;
    Gaussian_Elimination(2000);
    cout<<endl;

    cout<<"问题规模：2500"<<endl;
    Gaussian_Elimination(2500);
    cout<<endl;
}