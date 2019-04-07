#include<iostream>
#include<cmath>
#include<ctime>
#include "optimization.h"
//当权值变化小于MIN_DELT时，停止迭代
#define MIN_DELTA 0.000000000001
using namespace std;

Optimization::Optimization()
{
    featureNum = trainingNum = iterationNum = 0;
    learningRate = 0.0;
    dataset = NULL;
    weight = NULL;
}

Optimization::Optimization(int feature, int sample, int iteration, double rate, double **data, double *w)
{
    featureNum = feature;
    trainingNum = sample;
    iterationNum = iteration;
    learningRate = rate;
    dataset = data;
    weight = w;
}

Optimization::~Optimization()
{
    delete [] weight;
}

//函数指针指向计算不同梯度的函数(BGD/SGD/MBGD...)
void Optimization::gradientDescent(void (*gradient)(double** data, int features, int samples, double* w, double* gradient))
{
    clock_t start, finish;
    //计时；
    double timeCost=0.0;
    start=clock();
    //梯度
    double* grad = new double[featureNum];
    //权值
    weight = new double[featureNum];
    for(int i=0; i<featureNum; i++)
    {
        //初始化权值
        weight[i] = 0.0;
    }
    int iterations=iterationNum;
    while(iterations--)
    {
        //计算梯度
        (*gradient)(dataset, featureNum, trainingNum, weight, grad);
        double delta=0;
        for(int i=0; i<featureNum; i++)
        {
            delta+=(learningRate*grad[i])*(learningRate*grad[i]);
        }
        //计算权值变化量
        delta=sqrt(delta);
        //当权值变化小于MIN_DELT时，停止迭代，并输出迭代次数
        if(delta<=MIN_DELTA)
        {
            cout<<"There is almost no change in weight!\nThe iteration process is over!\n";
            cout<<"delta:"<<delta<<endl;
            cout<<"Iterations:"<<iterationNum - iterations<<endl;
            break;
        }
        for(int i=0; i<featureNum; i++)
        {
            //更新权值 w(k+1)=w(k)-r*gradient
            weight[i] -= learningRate*grad[i];
        }
    }
    delete [] grad;
    finish=clock();
    timeCost=(long double)(finish-start)*1000/CLOCKS_PER_SEC;
    cout<<"Time costs: "<<timeCost<<"ms"<<endl;
}

//函数指针指向计算不同梯度的函数(BGD/SGD/MBGD...)
void Optimization::Momentum(void (*gradient)(double** data, int features, int samples, double* w, double* gradient))
{
    clock_t start, finish;
    double timeCost=0.0;
    start=clock();
    double* grad = new double[featureNum];
    weight = new double[featureNum];
    //冲量
    double* mom=new double [featureNum];
    //冲量相关超参数beta，常设为0.9
    double beta=0.9;
    for(int i=0; i<featureNum; i++)
    {
        weight[i] = 0.0;
        mom[i]=0.0;
    }
    int iterations=iterationNum;
    double totalGrad=0.0001;
    while(iterations--)
    {
        (*gradient)(dataset, featureNum, trainingNum, weight, grad);
        for(int i=0; i<featureNum; i++)
        {
            //更新冲量 m(k+1)=beta*m(k) + rate*gradient
            mom[i] = beta*mom[i] + learningRate*grad[i];
            //更新权值 w(k+1)=w(k)-m(k+1)
            weight[i] -= mom[i];
        }
        double delta=0;
        for(int i=0; i<featureNum; i++)
        {
            delta+=(mom[i])*(mom[i]);
        }
        delta=sqrt(delta);
        if(delta<=MIN_DELTA)
        {
            cout<<"There is almost no change in weight!\nThe iteration process is over!\n";
            cout<<"delta:"<<delta<<endl;
            cout<<"Iterations:"<<iterationNum - iterations<<endl;
            break;
        }
    }
    delete [] grad;
    finish=clock();
    timeCost=(long double)(finish-start)*1000/CLOCKS_PER_SEC;
    cout<<"Time costs: "<<timeCost<<"ms"<<endl;
}

//函数指针指向计算不同梯度的函数
void Optimization::RMSprop(void (*gradient)(double** data, int features, int samples, double* w, double* gradient))
{
    clock_t start, finish;
    double timeCost=0.0;
    start=clock();
    double* grad = new double[featureNum];
    weight = new double[featureNum];
    //RMSprop相关超参数
    double beta=0.999;
    for(int i=0; i<featureNum; i++)
    {
        weight[i] = 0.0;
    }
    int iterations=iterationNum;
    //累计之前迭代过程中的梯度平方 s=beta*s+(1-beta)*||gradient||*||gradient||
    double totalGrad=0.00000001;
    while(iterations--)
    {
        (*gradient)(dataset, featureNum, trainingNum, weight, grad);
        double total=0.0;
        for(int i=0; i<featureNum; i++)
        {
            total+=grad[i]*grad[i];
        }
        totalGrad=beta*totalGrad+(1-beta)*total;
        double delta=0;
        for(int i=0; i<featureNum; i++)
        {
            delta+=(learningRate*grad[i]/sqrt(totalGrad))*(learningRate*grad[i]/sqrt(totalGrad));
        }
        delta=sqrt(delta);
        if(delta<=MIN_DELTA)
        {
            cout<<"There is almost no change in weight!\nThe iteration process is over!\n";
            cout<<"delta:"<<delta<<endl;
            cout<<"Iterations:"<<iterationNum - iterations<<endl;
            break;
        }
        for(int i=0; i<featureNum; i++)
        {
            //更新权值w(k+1)=w(k)-rate*gradient/sqrt(s)
            weight[i] -= learningRate*grad[i]/sqrt(totalGrad);
        }
    }
    delete [] grad;
    finish=clock();
    timeCost=(long double)(finish-start)*1000/CLOCKS_PER_SEC;
    cout<<"Time costs: "<<timeCost<<"ms"<<endl;
}

void Optimization::show() const
{
    cout<<"Feature Number:"<<featureNum<<endl;
    cout<<"Sample Number:"<<trainingNum<<endl;
    cout<<"Iteration Rate:"<<learningRate<<endl;
    cout<<"Iterations:"<<iterationNum<<endl;
    if(dataset==NULL)
        cout<<"No data"<<endl;
    else
    {
        cout<<"Dataset:"<<endl;
        for(int i=0;i<trainingNum;i++)
        {
            for(int j=0;j<featureNum+1;j++)
            {
                cout<<dataset[i][j]<<'\t';
            }
        cout<<endl;
        }
    }
    if(weight==NULL)
        cout<<"No weight"<<endl;
    else
    {
       for(int i=0;i<featureNum;i++)
        {
        cout<<"weight"<<i+1<<':'<<weight[i]<<endl;
        }
    }
    cout<<endl;
}

void Optimization::showWeight() const
{
    if(weight==NULL)
        cout<<"No weight"<<endl;
    else
    {
       for(int i=0;i<featureNum;i++)
        {
        cout<<"weight"<<i+1<<':'<<weight[i]<<endl;
        }
    }
    cout<<endl;
}