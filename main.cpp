#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define INPUT 784
#define OUTPUT 10
#define BATCH_SIZE 32
#define EPOCHS 50
#define LEARNING_RATE 0.1
#define LAMBDA 1e-4

using namespace std;

vector < vector < float> > test_images;
vector < vector < float> > train_images;
vector < vector < float> > test_labels;
vector < vector < float> > train_labels;


void readImageCSV(const string& name, vector< vector <float> > &vect, int size)
{
    string row, str;  int i = 0;

    ifstream file (name);
    if (file.is_open())
    {
        while (!file.eof())
        {
            if (i >= size)
                break;
            vect.emplace_back();
            getline(file, row);
            for (char c : row)
            {
                if (c != ',')
                {
                    str += c;
                }
                else
                {
                    vect[i].push_back(stof(str)/255.0);
                    str = "";
                }
            }
            if (!str.empty())
            {
                vect[i].push_back(stof(str)/255.0);
                str = "";
            }
            i++;
        }
    }
    else
        cout<<"Error opening file!"<<endl;
    file.close();
}

void readLabelCSV(const string& name, vector< vector <float> >  &vect, int size)
{
    string row;
    int  i = 0;
    vector < float> empty(OUTPUT, 0.0);

    ifstream file (name);
    if (file.is_open())
    {
        while (!file.eof())
        {
            if (i >= size)      //some ugly character was popping up after the last label
                break;
            getline(file, row);
            vect.push_back(empty);
            vect[i][stoi(row)] = 1.0;
            i++;
        }
    }
    else
        cout<<"Error opening file!"<<endl;
    file.close();
}

struct Gradients{
    vector < vector<float> > weight;
    vector< float> bias;
};

class Layer{
private:
public:
    Gradients gradients[BATCH_SIZE];
    vector <float> outputs[BATCH_SIZE];
    vector< vector< float> > weights;
    vector< float> bias;
    vector< float> output;
    int activation;              // 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax', else:'none'

    Layer(int input, int neurons, int act){
        srand(time(0));

        weights.reserve(neurons);
        output.reserve(neurons);
        activation = act;

        default_random_engine generator;
        normal_distribution<float> distribution(0.0,(2.0/(float)neurons));
        if (act == 2 || act == 3)
        {
            float range = sqrt(6.0 / (float)((input+neurons)*(input+neurons)));
            random_device rd;
            mt19937 generator(rd()); // seed the generator
            uniform_real_distribution<float> distribution(-range, range);
        }

        for (int neuron = 0; neuron < neurons; neuron++)
        {
            weights.emplace_back();
            bias.push_back(distribution(generator));
            output.push_back(0.0);

            for(int in = 0; in < input; in++)
            {
                weights[neuron].push_back(distribution(generator));
            }
        }
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            gradients[i].weight = weights;
            gradients[i].bias = bias;
            outputs[i] = output;
        }
    };

    float sigm(float i)
    {
        return (1 / (1 + exp(-i)));
    };

    float relu(float i)
    {
        return (i > 0.0 ? i : 0.0);
    };

    void softmax(vector<float> &x)
    {
        float divisor = 0;
        auto greatest = *max_element(x.begin(), x.end());
        for (auto & n : x)
        {
            n = exp(n - greatest);
            divisor += n;
        }
        for (auto & n : x)
        {
            n /= divisor;
        }
    };

    void activate(vector<float> &x)
    {
        switch(activation)
        {
            case 1 :
                for(float & i : x)
                    i = relu(i);
                break;
            case 2 :
                for(float & i : x)
                    i = sigm(i);
                break;
            case 3:
                for(float & i : x)
                    i = tanh(i);
                break;
            case 4:
                softmax(x);
                break;
            default:
                break;
        }
    };

    float de_relu(float i)
    {
        return (i > 0.0 ? 1.0 : 0.0);
    };
    float de_sigm(float i)
    {
        return (sigm(i) * (1 - sigm(i)));
    };
    float de_tanh(float i)
    {
        return (1 - (tanh(i)*tanh(i)));
    };

    void de_activate(vector<float> &x)
    {
        switch(activation)
        {
            case 1 :
                for(float & i : x)
                    i = de_relu(i);
                break;
            case 2 :
                for(float & i : x)
                    i = de_sigm(i);
                break;
            case 3:
                for(float & i : x)
                    i = de_tanh(i);
                break;
            case 4:
                break;
            default:
                for(float & i : x)
                    i = 1.0;
                break;
        }
    };

    vector<float> forward_pass(vector <float > &previous, bool train, uint32_t sample){

        fill(output.begin(), output.end(), 0.0);

        for (uint32_t neuron = 0; neuron < weights.size(); neuron++)
        {
            for(uint32_t input = 0; input < weights[neuron].size(); input++)
            {
                output[neuron] += (previous[input] * weights[neuron][input]);
            }
            output[neuron] = output[neuron] + bias[neuron] ;
        }

        activate(output);

        if (train){
            outputs[sample] = output;
        }
        return output;
    };

    vector<float> backward_pass(vector<float> &propagate, vector<float> &previous, uint32_t sample){
        vector<float> prop(weights[0].size() , 0.0);

        de_activate(outputs[sample]);
        for (uint32_t neuron = 0; neuron < weights.size(); neuron++)
        {
            gradients[sample].bias[neuron] = propagate[neuron] * outputs[sample][neuron];
            for (uint32_t input = 0; input < weights[neuron].size(); input++)
            {
                gradients[sample].weight[neuron][input] = gradients[sample].bias[neuron] * previous[input];
                prop[input] += gradients[sample].bias[neuron] * weights[neuron][input];
            }
        }
        return prop;
    };

    void gradient_descent(float learning_rate)
    {
        for(uint32_t sample = 0; sample < BATCH_SIZE; sample++)
        {
            for (uint32_t neuron = 0; neuron < weights.size(); neuron++)
            {
                bias[neuron] -= ((LAMBDA*bias[neuron] + gradients[sample].bias[neuron])/BATCH_SIZE)*learning_rate;
                for (uint32_t input = 0; input < weights[neuron].size(); input++)
                {
                    weights[neuron][input] -= ((LAMBDA*weights[neuron][input] + gradients[sample].weight[neuron][input])/BATCH_SIZE)*learning_rate;
                }
            }
        }
    };
};

class Network{
private:
    uint32_t n_layers;
    int last;
public:
    vector<Layer> layers;

    Network()
    {
        n_layers = -1;
        last = INPUT;
    };

    void add_layer(int neurons, int act)
    {
        Layer layer(last, neurons, act);
        layers.push_back(layer);
        last = neurons;
        n_layers++;
    };


    vector<float> MSE(vector<float> &x, vector<float> &y)
    {
        vector<float> error(x.size(), 0.0);
        for (int i = 0; i < x.size(); i++)
        {
            error[i] = x[i] - y[i];
        }
        return error;
    }

    vector< vector<float> > predict(vector < vector <float> > &x)
    {
        vector< vector<float> > prediction;
        prediction.reserve(x.size());

        for (uint32_t sample = 0; sample < x.size(); sample++)
        {
            vector< float> output(layers[0].weights.size() , 0.0);

            output = layers[0].forward_pass(x[sample], false, 0);

            for (uint32_t layer = 1; layer < layers.size(); layer++)
            {
                output = layers[layer].forward_pass(output, false, 0);
            }

            prediction.push_back(output);
        }
        return prediction;
    };

    void forward(vector<float> &sample, uint32_t index){
        vector< float> output(layers[0].weights.size() , 0.0);

        output = layers[0].forward_pass(sample, true, index);

        for (uint32_t layer = 1; layer < layers.size(); layer++)
        {
            output = layers[layer].forward_pass(output, true, index);
        }
    };

    void backward(vector <float> &x, vector<float> &y, uint32_t sample)
    {
        vector<float> error(layers[n_layers].weights[0].size() ,0.0);

        error = MSE(layers[n_layers].outputs[sample], y);

        for (int layer = n_layers; layer > 0; layer--)
        {
            error = layers[layer].backward_pass(error, layers[layer-1].outputs[sample], sample);
        }
        error = layers[0].backward_pass(error, x, sample);
    };

    float validate(vector < vector <float> > &predicted, vector <vector<float>> &truth)
    {
        int good = 0; int bad = 0;
        for (int i = 0; i < predicted.size(); i++)
        {
            auto predArgmax = distance(predicted[i].begin(), max_element(predicted[i].begin(), predicted[i].end()));
            auto trueArgmax = distance(truth[i].begin(), max_element(truth[i].begin(), truth[i].end()));
            if ( predArgmax == trueArgmax)
                good++;
            else
                bad++;
        }
        return (float)good/(float)predicted.size();

    };

    void train(vector < vector <float> > &x, vector <vector<float>> &y,
               vector < vector <float> > &xtest, vector <vector<float>> &ytest, int epochs, float learning_rate)
    {
        cout <<"train size : "<<x.size();
        auto n_batches = x.size() / BATCH_SIZE;
        auto remainder = x.size() % BATCH_SIZE;
        cout<<"  batches = "<<n_batches<<endl;

        /*cout<<endl<<"Untrained TRAIN ACC: "<<endl;
        auto predictions = predict(x);
        auto acc = validate(predictions, y);

        cout<<endl<<"Untrained VAL ACC: "<<endl;
        auto valpredictions = predict(xtest);
        acc = validate(valpredictions, ytest);*/

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int increment = 0;
            cout<<endl<<"learning rate = "<<learning_rate<<endl;
            for (int n = 0; n < n_batches; n++)
            {
                if (increment+BATCH_SIZE > BATCH_SIZE)
                    increment -= remainder;
                for (uint32_t sample = 0; sample < BATCH_SIZE; sample++)
                {
                    forward(x[sample+increment], sample);
                    backward(x[sample+increment], y[sample+increment], sample);
                }
                for (uint32_t layer=0; layer < layers.size(); layer++){
                    layers[layer].gradient_descent(learning_rate);
                }
                increment += BATCH_SIZE;
            }
            /*for (auto & layer : layers)
                for (auto & neuron : layer.weights)
                    cout<<*max_element(neuron.begin(), neuron.end())<<" ";*/
            cout<<"TRAIN ACC after epoch "<<epoch+1<<": "<<endl;
            auto predictions = predict(x);
            auto trainacc = validate(predictions, y);
            cout << trainacc*100.0<<"%"<<endl;

            cout<<endl<<"VAL ACC after epoch "<<epoch+1<<": "<<endl;
            auto valpredictions = predict(xtest);
            auto valacc = validate(valpredictions, ytest);
            cout << valacc*100.0<<"%"<<endl;

            if (valacc >=0.88)
            {
                export_csv(valpredictions);
                break;
            }

            learning_rate *= 0.98;
        }
    };
    void export_csv(vector<vector<float>> predicted)
    {
        ofstream file("actualPredictions");
        for (int i = 0; i < predicted.size(); i++)
        {
            auto predArgmax = distance(predicted[i].begin(), max_element(predicted[i].begin(), predicted[i].end()));
            file<<predArgmax<<endl;
        }
        file.close();
    };
};

int main(){

    train_images.reserve(TRAIN_SIZE);
    train_labels.reserve(TRAIN_SIZE);
    test_images.reserve(TEST_SIZE);
    test_labels.reserve(TEST_SIZE);

    readImageCSV("./data/fashion_mnist_train_vectors.csv", train_images, TRAIN_SIZE);
    readLabelCSV("./data/fashion_mnist_train_labels.csv", train_labels, TRAIN_SIZE);
    readImageCSV("./data/fashion_mnist_test_vectors.csv", test_images, TEST_SIZE);
    readLabelCSV("./data/fashion_mnist_test_labels.csv", test_labels, TEST_SIZE);

    cout << "files parsed "<<endl;

    Network model;

    model.add_layer(256, 1);
    model.add_layer(64, 1);
    model.add_layer(OUTPUT, 4);     // 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax', else:'none'

    model.train(train_images, train_labels, test_images, test_labels, EPOCHS, LEARNING_RATE);

    return 0;
}
