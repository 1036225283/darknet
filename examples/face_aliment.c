#include "darknet.h"
#include "image.h"
#include "data.h"
#include "list.h"

void train_face_aliment(char *cfgfile, char *weightfile)
{
    char *train_images = "/home/javer/work/data_set/helen/trainset/train.txt";
    char *backup_directory = "/home/javer/work/data_set/helen/";
    char *prefix = "/home/javer/work/data_set/helen/trainset";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    int i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths_with_prefix(train_images,prefix);
    //int N = plist->size;

    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = FACE_ALIMENT;

    args.angle = net->angle;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


void test_face_aliment(char *filename, char *weightfile)
{
  
}


image read_img()
{
    int w = 416;
    int h = 416;
    float hue = 0;
    float saturation = 1;
    float exposure = 1;
    image orig = load_image_color("/home/javer/work/data_set/helen/trainset/10405146_1.jpg", 0, 0);
    image sized = make_image(w, h, orig.c);
    fill_image(sized, .5);

    float new_ar = orig.w /(float)orig.h;
    //float scale = rand_uniform(.25, 2);
    float scale = 1;

    float nw, nh;

    if(new_ar < 1){
        nh = scale * h;
        nw = nh * new_ar;
    } else {
        nw = scale * w;
        nh = nw / new_ar;
    }

    float dx =  (w - nw)/2;
    float dy =  (h - nh)/2;

    place_image(orig, nw, nh, dx, dy, sized);

    random_distort_image(sized, hue, saturation, exposure);
    //show_image(sized,"tttt.png",-1);
    return orig;
}

matrix read_pts(int is_print)
{
    char *labelpath = "/home/javer/work/data_set/helen/trainset/10405146_1.pts";
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    matrix matrix_pts = make_matrix(1,136);
    
    char* a = fgetl(file);
    char* b = fgetl(file);
    char* c = fgetl(file);
    if(is_print)printf("%s\n%s\n%s\n",a,b,c);
    int i=0;
    for(i = 0;i<68;i++){
        if(fscanf(file, "%f %f\n", &matrix_pts.vals[0][i*2],&matrix_pts.vals[0][i*2+1]) == 2){
            if(is_print)printf("%f,%f\n",matrix_pts.vals[0][i*2],matrix_pts.vals[0][i*2+1]);
        } 
        else{
            if(is_print)printf("read %d line error\n",i);
        }
    }
    char* d = fgetl(file);
    if(is_print)printf("%s\n",d);
    if(a)free(a);
    if(b)free(b);
    if(c)free(c);
    if(d)free(d);
    fclose(file);
    return matrix_pts;
}


void run_face_aliment(int argc, char **argv)
{
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "test")) test_face_aliment(cfg, weights);
    else if(0==strcmp(argv[2], "train")) train_face_aliment(cfg, weights);
}

