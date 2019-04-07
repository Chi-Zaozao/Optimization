class Optimization
{
private:
    int featureNum;//特征数
    int trainingNum;//训练样本数
    int iterationNum;//最大迭代次数
    double learningRate;//步长（学习率）
    double **dataset;//数据集（指向二维数组）
    double *weight;//权值
public:
    //类初始化
    Optimization();
    Optimization(int feature, int sample, int iteration, double rate, double **data, double *w);
    ~Optimization();
    //传统梯度下降法
    void gradientDescent(void (*gradient)(double** data, int features, int samples, double* w, double* gradient));
    //Momentum冲量法
    void Momentum(void (*gradient)(double** data, int features, int samples, double* w, double* gradient));
    //RMSprop法
    void RMSprop(void (*gradient)(double** data, int features, int samples, double* w, double* gradient));  
    void show() const;
    //显示权值
    void showWeight() const;
};