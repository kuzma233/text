#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <arm_neon.h>
#include <sys/time.h>
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

void gaussian_NEON(float** A, float* b, int n) {
    for(int k = 0; k < n; k++){
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for(j = k+1; j+4 <= n; j+=4){
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for( ;j < n; j++){
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        b[k] = b[k]/A[k][k];
        for(int i = k+1; i < n; i++){
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for(j = k+1; j+4 <= n; j+=4){
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij , vx);
                vst1q_f32(&A[i][j] , vaij);
            }
            for( ; j < n; j++){
                A[i][j] = A[i][j]-A[i][k]*A[k][j];
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

void backup_NEON(float** A, float* b, float x[], int n) {
    x[n-1] = b[n-1] / A[n-1][n-1];
    float sum;
    for (int i = n - 1; i >= 0; --i) {
        sum = b[i];
        int j=0;
        for(j = i+1; j+4 < n; j+=4){
            float32x4_t vaij = vld1q_f32(&A[i][j]);
            float32x4_t vax = vld1q_f32(&x[j]);
            float32x4_t vx = vmulq_f32(vaij, vax);
            float32x2_t sumlow=vget_low_f32(vx);
            float32x2_t sumhigh=vget_high_f32(vx);
            sumlow=vpadd_f32(sumlow,sumhigh);
            float32_t addsum=vpadds_f32(sumlow);
            sum -= float(addsum);
        }
        for( ; j < n; j++){
            sum -= A[i][j]* x[j];
        }
        x[i] = sum / A[i][i];
    }
}

//主进程计时
void Gaussian_Elimination(int n){
    struct timeval start1,end1,start2,end2;

    float** A;
    float* b;
    float* x;
    x=new float[n];
    ini_Matrix(A,b,n);

    gettimeofday(&start1,NULL);
    gaussian(A, b, n);
    gettimeofday(&end1,NULL);
    
    gettimeofday(&start2,NULL);
    backup(A, b, x, n);
    gettimeofday(&end2,NULL);

    double interval1=(end1.tv_sec-start1.tv_sec)*1000.0+
    (end1.tv_usec - start1.tv_usec)/1000.0;
    double interval2=(end2.tv_sec-start2.tv_sec)*1000.0+
    (end2.tv_usec - start2.tv_usec)/1000.0;
    cout<<"-----------普通算法-------------"<<endl;
    cout <<"elimination:"<<interval1 << "ms" << endl;
    cout <<"backup:"<<interval2 << "ms" << endl;
    cout <<"Total time:"<<interval1+interval2 <<"ms" <<endl;
}

void Gaussian_Elimination_NEON(int n){
    struct timeval start1,end1,start2,end2;

    float** A;
    float* b;
    float* x;
    x=new float[n];
    ini_Matrix(A,b,n);

    gettimeofday(&start1,NULL);
    gaussian_NEON(A, b, n);
    gettimeofday(&end1,NULL);
    
    gettimeofday(&start2,NULL);
    backup_NEON(A, b, x, n);
    gettimeofday(&end2,NULL);

    double interval1=(end1.tv_sec-start1.tv_sec)*1000.0+
    (end1.tv_usec - start1.tv_usec)/1000.0;
    double interval2=(end2.tv_sec-start2.tv_sec)*1000.0+
    (end2.tv_usec - start2.tv_usec)/1000.0;
    cout<<"---------NEON并行优化算法-----------"<<endl;
    cout <<"elimination:"<<interval1 << "ms" << endl;
    cout <<"backup:"<<interval2 << "ms" << endl;
    cout <<"Total time:"<<interval1+interval2 <<"ms" <<endl;
}

int main() {
    cout<<"问题规模：500"<<endl;
    Gaussian_Elimination(500);
    Gaussian_Elimination_NEON(500);
    cout<<endl;
    
    cout<<"问题规模：1000"<<endl;
    Gaussian_Elimination(1000);
    Gaussian_Elimination_NEON(1000);
    cout<<endl;

    cout<<"问题规模：2000"<<endl;
    Gaussian_Elimination(2000);
    Gaussian_Elimination_NEON(2000);
    cout<<endl;

    cout<<"问题规模：3000"<<endl;
    Gaussian_Elimination(3000);
    Gaussian_Elimination_NEON(3000);
    cout<<endl;

    cout<<"问题规模：4000"<<endl;
    Gaussian_Elimination(4000);
    Gaussian_Elimination_NEON(4000);
    cout<<endl;

}
