#include<iostream>
#include<fstream>
#include<random>
#include"optimization.h"
//小批量梯度下降法batch size设置
#define BATCHSIZE 5
using namespace std;
//全量梯度下降法
void BGD(double** data, int features, int samples, double* w, double* gradient);
//随机梯度下降法
void SGD(double** data, int features, int samples, double* w, double* gradient);
//小批量梯度下降
void MBGD(double** data, int features, int samples, double* w, double* gradient);

int main()
{
    ifstream inFile;
    //读入数据
    inFile.open("data.txt");
    if(!inFile.is_open())
    {
        cout<<"Could not open the file!"<<endl;
        cout<<"Program terminating..."<<endl;
        exit(EXIT_FAILURE);
    }
    int featureNum;
    int trainingNum;
    int iterationNum;
    double learningRate;
    //读入超参数
    inFile>>featureNum>>trainingNum>>learningRate>>iterationNum;
    double** dataset;
    dataset = new double*[trainingNum];
    for(int i=0; i<trainingNum; i++)
    {
        dataset[i] = new double[featureNum+1];
    }
    for(int i=0;i<trainingNum;i++)
    {
        for(int j=0;j<featureNum+1;j++)
        {
            //读入训练样本数据
            inFile>>dataset[i][j];
        }
    }
    inFile.close();
    double* weight = new double[featureNum];
    weight=NULL;
    Optimization op{featureNum,trainingNum,iterationNum,learningRate,dataset,weight};
    op.show();
    //传统梯度下降法（全量）
    op.gradientDescent(BGD);
    cout<<"BGD:"<<endl;
    op.showWeight();
     //传统梯度下降法（随机）
    op.gradientDescent(SGD);
    cout<<"SGD:"<<endl;
    op.showWeight();
     //传统梯度下降法（小批量）
    op.gradientDescent(MBGD);
    cout<<"MBGD:"<<endl;
    op.showWeight();
    //冲量法（全量）
    op.Momentum(BGD);
    cout<<"BGD(MOmentum):"<<endl;
    op.showWeight();
    //冲量法（随机）
    op.Momentum(SGD);
    cout<<"SGD(MOmentum):"<<endl;
    op.showWeight();
    //冲量法（小批量）
    op.Momentum(MBGD);
    cout<<"MBGD(MOmentum):"<<endl;
    op.showWeight();
    //RMSprop（全量）
    op.RMSprop(BGD);
    cout<<"BGD(RMSprop):"<<endl;
    op.showWeight();
    //RMSprop（随机）
    op.RMSprop(SGD);
    cout<<"SGD(RMSprop):"<<endl;
    op.showWeight();
    //RMSprop（小批量）
    op.RMSprop(MBGD);
    cout<<"MBGD(RMSprop):"<<endl;
    op.showWeight();
    for(int i=0; i<trainingNum; i++)
    {
        delete [] dataset[i];
    }
    delete [] dataset;
    cout<<"Press any key to stop..."<<endl;
    cin.get();
    return 0;
}

void BGD(double** data, int features, int samples, double* w, double* gradient)
{
    double* polynomial = new double[samples];
    //利用所有样本计算梯度
    for(int j=0; j<samples; j++)
    {
        double sum=0.0;
        for(int t=0; t<features; t++)
        {
             sum+=w[t]*data[j][t];
        }
        //计算每个样本的多项式
        polynomial[j] = sum - data[j][features];
    }
    for(int i=0; i<features; i++)
    {
        gradient[i] = 0.0;
        for(int j=0; j<samples; j++)
        {
            //梯度=各多项式分别乘以样本后求和
            gradient[i]+=polynomial[j]*data[j][i];
        }
    }
    delete [] polynomial;
}

void SGD(double** data, int features, int samples, double* w, double* gradient)
{
    static default_random_engine e; 
	static uniform_int_distribution<> dis(1, samples);
    //随机选取一个样本计算梯度
    int randSample = dis(e)-1;
    double sum=0.0;
    for(int t=0; t<features; t++)
    {
        sum+=w[t]*data[randSample][t];
    }
    double polynomial = sum - data[randSample][features];
    for(int i=0; i<features; i++)
    {
        gradient[i]=polynomial*data[randSample][i];
    }
}

void MBGD(double** data, int features, int samples, double* w, double* gradient)
{
    static default_random_engine e; 
	static uniform_int_distribution<> dis(1, samples);
    int batchSample[BATCHSIZE]={0};
    //随机选取Batch size个不同的样本计算梯度
    //Batch size=1时，该方法同随机梯度下降法
    //Batch size=样本数 时，该方法同全量梯度下降法
    for(int i=0;i<BATCHSIZE; i++)
    {
        bool equal = true;
        while(equal)
        {
            equal = false;
            batchSample[i]=dis(e)-1;
            for(int j=0; j<i; j++)
            {
                //保证随机选取的各个样本不相同
                if(batchSample[i]==batchSample[j])
                {
                    equal=true;
                    break;
                }
            }
        }    
    }
    double* polynomial = new double[BATCHSIZE];
    for(int j=0; j<BATCHSIZE; j++)
    {
        double sum=0.0;
        for(int t=0; t<features; t++)
        {
             sum+=w[t]*data[batchSample[j]][t];
        }
        polynomial[j] = sum - data[batchSample[j]][features];
    }
    for(int i=0; i<features; i++)
    {
        gradient[i] = 0.0;
        for(int j=0; j<BATCHSIZE; j++)
        {
            gradient[i]+=polynomial[j]*data[batchSample[j]][i];
        }
    }
    delete [] polynomial;
}
