#include <iostream>
#include <ctime>  //随机种子
#include <windows.h> //计时
#include <immintrin.h> //AVX、AVX2
#include <omp.h>  //openmp
#include <mpi.h>   //MPI
#define NUM_THREADS 7
using namespace std;

// 初始化矩阵
void ini_Matrix(float** &A, int n) {
    srand(time(NULL));
    // 动态分配
    A = new float*[n];
    for (int i = 0; i < n; ++i) {
        A[i] = new float[n];
        for (int j = 0; j < n; ++j) {
            A[i][j] = float(rand() % 10 + 1); // 生成1到10的随机数填充矩阵
        }
    }
}
void ini_Empty(float** &A, int n) 
{
    A = new float* [n];
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        memset(A[i], 0, n*sizeof(float));
    }
}

// 高斯消去
void gaussian(float **A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float factor = A[j][i] / A[i][i];
            for (int k = i; k < n; ++k) {
                A[j][k] -= factor * A[i][k];
            }
        }
    }
}
//主进程计时
void Gaussian_Elimination(int n){
    long long freq,start_0,end_0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    float** A;
    ini_Matrix(A,n);
    QueryPerformanceCounter((LARGE_INTEGER*)&start_0);
    gaussian(A,n);
    QueryPerformanceCounter((LARGE_INTEGER*)&end_0);
    double interval_0 = (end_0 - start_0) * 1000.0 / freq;
    cout <<"串行算法 :"<<interval_0 <<" ms" <<endl;
}


void LU_mpi(int argc, char* argv[],int N) {  //块划分
    float** A=NULL;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        ini_Matrix(A,N);
        for (int j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (int i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }
    }
    else {
        ini_Empty(A,N);
        for (int i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  //此时每个进程都拿到了数据
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (int j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < total; j++) { 
                if(j!=rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0号消息表示除法完毕
            }
        }
        else {
            int src;
            if (k < N / total * total)//在可均分的任务量内
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (int i = begin; i < end; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == 0) {//0号进程中存有最终结果
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
}

void LU_mpi_circle(int argc, char* argv[],int N) {  //等步长的循环划分
    float** A=NULL;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {  //0号进程初始化矩阵
        ini_Matrix(A,N);
        for (int j = 1; j < total; j++) {
            for (int i = j; i < N; i += total) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }
    }
    else {
        ini_Empty(A,N);
        for (int i = rank; i < N; i += total) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++) {
        if (k % total == rank) {
            for (int j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int j = 0; j < total; j++) { //
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0号消息表示除法完毕
            }
        }
        else {
            int src = k % total;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % total != rank)
            begin++;
        for (int i = begin; i < N; i += total) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == 0) {//0号进程中存有最终结果
        end_time = MPI_Wtime();
        printf("平凡MPI,循环划分耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
}

void LU_mpi_async(int argc, char* argv[],int N) {  //非阻塞通信
    float** A=NULL;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        ini_Matrix(A,N);
        MPI_Request* request = new MPI_Request[N - end];
        for (int j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递
    }
    else {
        ini_Empty(A,N);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (int k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (int j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];   //非阻塞传递
            for (int j = rank + 1; j < total; j++) { 
                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j -rank - 1]);//0号消息表示除法完毕
            }
            MPI_Waitall(total -rank - 1, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //若执行完自身的任务，可直接跳出
        }
        else {
            int src;
            if (k < N / total * total)//在可均分的任务量内
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (int i = begin; i < end; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分+非阻塞耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
}

void LU_mpi_async_multithread(int argc, char* argv[],int N) {  //非阻塞通信+avx+omp
    float** A=NULL;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        ini_Matrix(A,N);
        MPI_Request* request = new MPI_Request[N - end];
        for (int j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (int i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递
    }
    else {
        ini_Empty(A,N);
        MPI_Request* request = new MPI_Request[end - begin];
        for (int i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                __m256 t1 = _mm256_set1_ps(A[k][k]);
                int j = 0;
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&A[k][j]);  //AVX优化除法部分
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&A[k][j], t2);
                }
                for (; j < N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
                for (j = 0; j < total; j++) { 
                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)//在可均分的任务量内
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided)  //开始多线程
        for (int i = begin; i < end; i++) {
            __m256 vik = _mm256_set1_ps(A[i][k]);   //AVX优化消去部分
            int j = 0;
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
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分+非阻塞+OpenMP+AVX耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
}


int main(int argc, char* argv[])
{
    // Gaussian_Elimination(3000);
    // LU_mpi(argc,argv,3000);
    // LU_mpi_circle(argc,argv,3000);
    // LU_mpi_async(argc,argv,3000);
    LU_mpi_async_multithread(argc,argv,1000);
}