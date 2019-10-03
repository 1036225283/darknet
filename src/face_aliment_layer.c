#include "face_aliment_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

float mean_face[136] = {
1023.794428,1262.260007,
1028.649637,1408.555036,
1058.086618,1567.417124,
1100.665855,1711.323493,
1150.771725,1841.138392,
1248.015343,1958.747616,
1372.482579,2028.915945,
1520.527579,2090.654244,
1664.069203,2101.536861,
1780.385135,2076.603600,
1885.288546,2000.783728,
1979.413734,1907.265492,
2051.327923,1791.564848,
2093.904804,1658.227211,
2117.334545,1527.048855,
2119.028760,1395.223971,
2104.999168,1271.927012,
1166.046497,1152.195936,
1270.982085,1084.009022,
1397.482401,1089.236504,
1496.992053,1112.163732,
1592.512618,1171.399899,
1700.446373,1136.086927,
1792.008865,1119.582878,
1892.511528,1100.731476,
1989.516805,1089.605561,
2063.515018,1157.394658,
1664.132005,1263.748118,
1671.530823,1353.238678,
1682.813619,1433.791134,
1693.156859,1520.556973,
1555.986669,1597.325911,
1614.709215,1617.626457,
1688.170332,1631.380945,
1745.097309,1613.796186,
1797.093593,1595.885721,
1282.566340,1270.845697,
1349.930418,1238.685238,
1416.812382,1245.136397,
1501.432342,1297.108529,
1416.692189,1284.499279,
1345.329253,1286.291016,
1765.572596,1299.455812,
1839.801727,1239.481371,
1905.237939,1238.579485,
1963.181392,1273.394862,
1909.651134,1287.959704,
1842.944854,1285.967751,
1434.876537,1766.365514,
1525.995591,1722.063362,
1627.353637,1708.447051,
1690.902765,1720.254150,
1758.477793,1704.567840,
1815.466145,1713.760411,
1859.693423,1754.681193,
1810.013301,1804.195551,
1755.143450,1837.958060,
1685.454997,1853.175828,
1624.457908,1849.804158,
1528.358647,1814.349189,
1465.790403,1754.487015,
1624.218746,1754.559493,
1691.420991,1762.073456,
1753.969071,1750.918648,
1833.323074,1745.029558,
1753.969071,1750.918648,
1690.691640,1781.866917,
1623.333048,1769.015456,
};

void norm_mean_face(int w,int h)
{
    for(int i=0;i<68;i++){
        mean_face[i*2] = mean_face[i*2]/w;
        mean_face[i*2+1] = mean_face[i*2+1]/h;
    }
}

layer make_face_aliment_layer(int batch, int w, int h, int n)
{
    layer l = {0};
    l.type = FACE_ALIMENT;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*141;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*n*141;
    l.inputs = l.outputs;
    l.truths = 68*2;
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_face_aliment_layer;
    l.backward = backward_face_aliment_layer;
#ifdef GPU
    l.forward_gpu = forward_face_aliment_layer_gpu;
    l.backward_gpu = backward_face_aliment_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "face_aliment\n");
    srand(0);
    norm_mean_face(3172,2828);

    return l;
}

void resize_face_aliment_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

static box get_face_detect_box(float *x, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) / w;
    b.h = exp(x[index + 3*stride]) / h;
    return b;
}

static float delta_face_detect_box(box truth, float *x, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_face_detect_box(x, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w);
    float th = log(truth.h*h);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


static float logit(float x)
{
    return log(x/(1.-x));
}

static float tisnan(float x)
{
    return (x != x);
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*5 + entry*l.w*l.h + loc;
}

static box get_truth_box(float *y,int points)
{
    float min_x = 1.0,max_x = 0.0,min_y = 1.0,max_y = 0.0;
    int i = 0;
    for(i=0;i<points;i++){
        if(y[i*2]>max_x){
            max_x = y[i*2];
        }
        if(y[i*2]<min_x){
            min_x = y[i*2];
        }
        if(y[i*2+1]>max_y){
            max_y = y[i*2+1];
        }
        if(y[i*2+1]<min_y){
            min_y = y[i*2+1];
        }
    }
    box b;
    b.x = min_x + (max_x-min_x)/2.0f;
    b.y = min_y + (max_y-min_y)/2.0f;
    b.w = max_x - min_x;
    b.h = max_y - min_y;
    b.w += b.w/20;
    b.h += b.h/20;
    if(b.w <= 0 || b.h <= 0 || max_x <= 0 || max_y<=0){
        printf("%f,%f,%f,%f,%f,%f,%f,%f\n",min_x,min_y,max_x,max_y,b.x,b.y,b.w,b.h);
    }
    assert(b.w>0&&b.h>0);
    return b;
}

static float delta_face_landmars(box fbox, float *truth, float *pred, float *delta, int index, int stride, float scale)
{
    int i;
    float diff = 0;
    for(i=0;i<68;++i){
        float tx = truth[i*2]-fbox.x;
        float ty = truth[i*2+1]-fbox.y;
        //printf("tx=%f,ty=%f,box: %f,%f,%f,%f,truth: %f,%f\n",tx,ty,fbox.x,fbox.y,fbox.w,fbox.h,truth[i*2],truth[i*2+1]);
        delta[index + (i*2)*stride] = scale*(tx - pred[index + (i*2)*stride]);
        delta[index + (i*2+1)*stride] = scale*(ty - pred[index + (i*2+1)*stride]);
        diff += 0.5*delta[index + (i*2)*stride]*delta[index + (i*2)*stride] + 0.5*delta[index + (i*2+1)*stride]*delta[index + (i*2+1)*stride];
    }
    return diff;
}

void forward_face_aliment_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            //index = entry_index(l, b, n*l.w*l.h, 5);
            //activate_array(l.output + index,   l.truths*l.w*l.h, TANH);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    float avg_landmark_diff = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_face_detect_box(l.output, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    box truth = get_truth_box(net.truth+68*2*b,68);
                    if(!truth.x) break;
                    float iou = box_iou(pred, truth);
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.noobject_scale*(0 - l.output[obj_index]);
                    if (iou > 0.5) {
                        l.delta[obj_index] = 0;
                    }
                    if(*(net.seen) < 30000){
                        box mean = get_truth_box(mean_face,68);
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = mean.w;
                        truth.h = mean.h;
                        int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        delta_face_detect_box(truth, l.output, n, box_index, i, j, l.w, l.h, l.delta, 0.1, l.w*l.h);
                        int landmarks_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 5);
                        delta_face_landmars(truth, mean_face, l.output, l.delta, landmarks_index,l.w*l.h,0.1);
                    }
                }
            }
        }

        box truth = get_truth_box(net.truth+b*l.truths,68);
        if(!truth.x) break;
        float best_iou = 0;
        int best_n = 0;
        i = (truth.x * l.w);
        j = (truth.y * l.h);
        box truth_shift = truth;
        truth_shift.x = 0;
        truth_shift.y = 0;
        for(n = 0; n < l.n; ++n){
            int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
            box pred = get_face_detect_box(l.output, n, box_index, i, j, l.w, l.h, l.w*l.h);
            pred.x = 0;
            pred.y = 0;
            float iou = box_iou(pred, truth_shift);
            if (iou > best_iou){
                best_iou = iou;
                best_n = n;
            }
        }

        int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
        float iou = delta_face_detect_box(truth, l.output, best_n, box_index, i, j, l.w, l.h, l.delta, 1, l.w*l.h);
        if(iou > .5) recall += 1;
        avg_iou += iou;

        int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
        avg_obj += l.output[obj_index];
        l.delta[obj_index] = l.object_scale*(1 - l.output[obj_index]);
        int landmarks_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 5);
        float diff = delta_face_landmars(truth, net.truth+b*l.truths, l.output, l.delta, landmarks_index,l.w*l.h,1);
        avg_landmark_diff += diff;
        ++count;
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Obj: %f, No Obj: %f, Avg Recall: %f, Avg Landmark: %f, count: %d\n", avg_iou/count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, avg_landmark_diff/count, count);
}

void backward_face_aliment_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}


void get_face_aliment_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n,z;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            int obj_index  = entry_index(l, 0, index, 4);
            int box_index  = entry_index(l, 0, index, 0);
            dets[index].bbox = get_face_detect_box(predictions, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = predictions[obj_index];
            dets[index].sort_class = 0;
            for(int i=0;i<68;i++){
                int landmark_index  = entry_index(l, 0, index, (2*i)+5);
                dets[index].aliment[2*i] = predictions[landmark_index];
                dets[index].aliment[2*i] += dets[index].bbox.x;
                dets[index].aliment[2*i] *= w;
                landmark_index  = entry_index(l, 0, index, (2*i+1)+5);
                dets[index].aliment[2*i+1] = predictions[landmark_index];
                dets[index].aliment[2*i+1] += dets[index].bbox.y;
                dets[index].aliment[2*i+1] *= h;
                //printf("%f,%f\n",dets[index].aliment[2*i],dets[index].aliment[2*i+1]);
            }
            dets[index].bbox.x *= w;
            dets[index].bbox.y *= h;
            dets[index].bbox.w *= w;
            dets[index].bbox.h *= h;
            dets[index].bbox.x -= dets[index].bbox.w/2;
            dets[index].bbox.y -= dets[index].bbox.h/2;
        }
    }
}

#ifdef GPU

void forward_face_aliment_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
            //index = entry_index(l, b, n*l.w*l.h, 5);
            //activate_array_gpu(l.output_gpu + index, l.truths*l.w*l.h, TANH);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_face_aliment_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_face_aliment_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            index = entry_index(l, b, n*l.w*l.h, 4);
            gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
            //index = entry_index(l, b, n*l.w*l.h, 5);
            //gradient_array_gpu(l.output_gpu + index, l.truths*l.w*l.h, TANH, l.delta_gpu + index);
        }
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif


