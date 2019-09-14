#include "darknet.h"

void train_mnist(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    char *backup_directory = "backup/";
    int N = 50000;

    int epoch = (*net->seen)/N;
    data train = load_mnist_data("data/mnist/train-images-idx3-ubyte","data/mnist/train-labels-idx1-ubyte",N);
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        clock_t time=clock();

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net->seen);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free(base);
    free_data(train);
}


void test_mnist(char *filename, char *weightfile)
{
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    clock_t time;
    float avg_acc = 0;
    float avg_top5 = 0;
    int N = 10000;
    data test = load_mnist_data("data/mnist/train-images-idx3-ubyte","data/mnist/train-labels-idx1-ubyte",N);

    time=clock();

    float *acc = network_accuracies(net, test, 2);
    avg_acc += acc[0];
    avg_top5 += acc[1];
    printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock()-time), test.X.rows);
    free_data(test);
}


void run_mnist(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_mnist(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_mnist(cfg, weights);
}
