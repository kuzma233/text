#include<iostream>
#include <ctime>  //随机种子
#include <sys/time.h> //计时
#include <arm_neon.h> //NEON
#include<pthread.h> //pthread
#include<semaphore.h>
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

//AVX & pthread
sem_t sem_main;  //信号量
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

struct threadParam_t {    //参数数据结构
    int k;
    int t_id;
    float** A;
    float* b;
    int n;
};

void* avx_threadFunc(void* param) {          //avx优化算法,动态分配
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务
    float** A=p->A;
    float* b=p->b;
    int N=p->n;

    float32x4_t vaik = vdupq_n_f32(A[i][k]);
    int j;
    for (j = k + 1; j + 4 <= N; j += 4) {
        float32x4_t vakj = vld1q_f32(&A[k][j]);
        float32x4_t vaij = vld1q_f32(&A[i][j]);
        float32x4_t vx = vmulq_f32(vakj, vaik);
        vaij = vsubq_f32(vaij, vx);
        vst1q_f32(&A[i][j], vaij);
    }
    for (; j < N; j++) {
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
    b[i] = b[i] - A[i][k]*b[k];
    A[i][k] = 0;
    
    pthread_exit(NULL);
    return NULL;
}
void avx_dynamic(float** A,float* b,int N) {            
    for (int k = 0; k < N; k++) {
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4) {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        b[k] = b[k]/A[k][k];
        A[k][k] = 1.0;
        
        int thread_cnt = N - 1 - k;
        pthread_t* handle = (pthread_t*)malloc(thread_cnt * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(thread_cnt * sizeof(threadParam_t));

        for (int t_id = 0; t_id < thread_cnt; t_id++) {//分配任务
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].A=A;
            param[t_id].b=b;
            param[t_id].n=N;
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++) {
            pthread_create(&handle[t_id], NULL, avx_threadFunc, &param[t_id]);
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++) {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }
}


void* sem_threadFunc(void* param) {    //静态分配8线程+信号量
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    float** A=p->A;
    float* b=p->b;
    int N=p->n;

    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workstart[t_id]);//阻塞，等待主线程除法完成

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j + 4 <= N; j += 4) {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k]*b[k];
            A[i][k] = 0.0;
        }

        sem_post(&sem_main);        //唤醒主线程
        sem_wait(&sem_workend[t_id]);  //阻塞，等待主线程唤醒进入下一轮

    }
    pthread_exit(NULL);
    return NULL;
}
void sem_static(float** A,float* b,int N) {
    sem_init(&sem_main, 0, 0); //初始化信号量
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }
    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        param[t_id].A=A;
        param[t_id].b=b;
        param[t_id].n=N;
        pthread_create(&handle[t_id], NULL, sem_threadFunc, &param[t_id]);
    }

    for (int k = 0; k < N; k++) {
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4) {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        b[k] = b[k]/A[k][k];
        A[k][k] = 1.0;

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {  //唤起子线程
            sem_post(&sem_workstart[t_id]);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {  //主线程睡眠
            sem_wait(&sem_main);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {  //再次唤起工作线程，进入下一轮消去
            sem_post(&sem_workend[t_id]);
        }

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handle[t_id], NULL);
    }
    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}


void* sem_triplecircle_thread(void* param) { //静态线程+信号量+三重循环
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    float** A=p->A;
    float* b=p->b;
    int N=p->n;

    for (int k = 0; k < N; k++) { //0号线程做除法，其余等待
        if (t_id == 0) {
            float32x4_t vt = vdupq_n_f32(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= N; j += 4) {
                float32x4_t va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            b[k] = b[k]/A[k][k];
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {   //主线程唤醒其余线程
                sem_post(&sem_Division[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j + 4 <= N; j += 4) {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k]*b[k];
            A[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}
void sem_triplecircle(float** A,float* b,int N) {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        param[t_id].A=A;
        param[t_id].b=b;
        param[t_id].n=N;
        pthread_create(&handle[t_id], NULL, sem_triplecircle_thread, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handle[t_id], NULL);
    }
    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}


void* barrier_threadFunc(void* param) {         //静态分配8线程，barrier同步
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    float** A=p->A;
    float* b=p->b;
    int N=p->n;

    for (int k = 0; k < N; k++) { //0号线程做除法
        if (t_id == 0) {
            float32x4_t vt = vdupq_n_f32(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= N; j += 4) {
                float32x4_t va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            b[k] = b[k]/A[k][k];
            A[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Division);//第一个同步点

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j + 4 <= N; j += 4) {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k]*b[k];
            A[i][k] = 0.0;
        }

        pthread_barrier_wait(&barrier_Elimination);//第二个同步点
    }
    pthread_exit(NULL);
    return NULL;
}
void barrier_static(float** A,float* b,int N)
{
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        param[t_id].A=A;
        param[t_id].b=b;
        param[t_id].n=N;
        pthread_create(&handle[t_id], NULL, barrier_threadFunc, &param[t_id]);

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handle[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);

    free(handle);
    free(param);
}



//主进程计时
void Gaussian_Elimination(int n){
    float** A;
    float* b;
    float* x;
    x=new float[n];
    
    struct timeval start_0,end_0,start_1,end_1,start_2,end_2,start_3,end_3,start_4,end_4;

    ini_Matrix(A,b,n);
    gettimeofday(&start_0,NULL);
    gaussian(A, b, n);
    backup(A, b, x, n);
    gettimeofday(&end_0,NULL);

    ini_Matrix(A,b,n);
    gettimeofday(&start_1,NULL);
    avx_dynamic(A, b, n);
    backup(A, b, x, n);
    gettimeofday(&end_1,NULL);

    ini_Matrix(A,b,n);
    gettimeofday(&start_2,NULL);
    sem_static(A, b, n);
    backup(A, b, x, n);
    gettimeofday(&end_2,NULL);

    ini_Matrix(A,b,n);
    gettimeofday(&start_3,NULL);
    sem_triplecircle(A, b, n);
    backup(A, b, x, n);
    gettimeofday(&end_3,NULL);

    ini_Matrix(A,b,n);
    gettimeofday(&start_4,NULL);
    barrier_static(A, b, n);
    backup(A, b, x, n);
    gettimeofday(&end_4,NULL);

    double interval_0=(end_0.tv_sec-start_0.tv_sec)*1000.0+
    (end_0.tv_usec - start_0.tv_usec)/1000.0;
    cout <<"串行算法 time:"<<interval_0 <<"ms" <<endl;
    double interval_1=(end_1.tv_sec-start_1.tv_sec)*1000.0+
    (end_1.tv_usec - start_1.tv_usec)/1000.0;
    cout <<"动态分配线程 time:"<<interval_1 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_1<<endl;
    double interval_2=(end_2.tv_sec-start_2.tv_sec)*1000.0+
    (end_2.tv_usec - start_2.tv_usec)/1000.0;
    cout <<"静态分配8线程，sem_t同步 time:"<<interval_2 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_2<<endl;
    double interval_3=(end_3.tv_sec-start_3.tv_sec)*1000.0+
    (end_3.tv_usec - start_3.tv_usec)/1000.0;
    cout <<"静态分配8线程，sem_t同步，加入三重循环 time:"<<interval_3 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_3<<endl;
    double interval_4=(end_4.tv_sec-start_4.tv_sec)*1000.0+
    (end_4.tv_usec - start_4.tv_usec)/1000.0;
    cout <<"静态分配8线程，barrier同步 time:"<<interval_4 <<"ms" <<endl;
    cout << "加速比："<<interval_0/interval_4<<endl;

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