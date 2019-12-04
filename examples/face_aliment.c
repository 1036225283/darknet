#include "darknet.h"
#include "../src/image.h"
#include "../src/data.h"
#include "../src/list.h"

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
    args.type = FACE_ALIMENT_DATA;

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

        //printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d, loss: %f, avg: %f, rate: %f, seconds: %lf, images: %d \n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 && i>0){
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


void test_face_aliment(char *filename, char *weightfile,char *pic_path)
{
    int w=448,h=448;
    char *base = basecfg(filename);
    printf("%s\n", base);
    network *net = load_network(filename, weightfile, 0);
    image orig = load_image_color(pic_path, 0, 0);
    image sized = letterbox_image(orig, net->w, net->h);

    float* result = network_predict(net,sized.data);

    //parse and show result
    int nboxes = 0;
    float thresh = 0.6;
    detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, 0, 0, 1, &nboxes);
    //printf("%d\n", nboxes);
    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    do_nms_sort_simple(dets, nboxes, 0.5);
    for(int i=0;i<nboxes;i++){
        //printf("%f\n",dets[i].objectness);
        if(dets[i].objectness > thresh){
            box tmpb = dets[i].bbox;
            draw_box(sized,tmpb.x, tmpb.y, tmpb.x+tmpb.w,tmpb.y+tmpb.h, 1.0, 0.0, 0.0);
            draw_face_landmark_with_truth(sized,dets[i].aliment,3,0,1.,0,5);
        }
    }
    
    free_detections(dets, nboxes);

#ifdef OPENCV
    make_window("predictions", w, h, 0);
    show_image(sized, "predictions", 0);
#endif
}



void run_face_aliment(int argc, char **argv)
{
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *pic_path = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "test")) test_face_aliment(cfg, weights, pic_path);
    else if(0==strcmp(argv[2], "train")) train_face_aliment(cfg, weights);
}

