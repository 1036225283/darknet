#include "face_aliment_layer.h"

face_aliment_layer make_face_aliment_layer(int batch, int w, int h, int n)
{
    layer l = {0};
    l.type = FACE_ALIMENT;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*n;
    l.inputs = l.outputs;
    l.truths = 136;
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

    fprintf(stderr, "face aliment\n");
    srand(0);

    return l;
}


void resize_face_aliment_layer(face_aliment_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n;
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

static int entry_index(layer l, int batch, int entry, int h_index, int w_index)
{
    return batch*l.outputs + entry*l.w*l.h + h_index*l.w + w_index;
}

static box get_region_box(float *x, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) / w;
    b.h = exp(x[index + 3*stride]) / h;
    return b;
}

static float delta_region_box(box truth, float *x, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, n, index, i, j, w, h, stride);
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

static box get_truth_box(float *y,int points)
{
    float min_x = 1,max_x = 0,min_y = 1,max_y = 0;
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
        if(y[i*2+1]<min_x){
            min_y = y[i*2+1];
        }
    }
    box b;
    b.x = min_x + (max_x-min_x)/2.0f;
    b.y = min_y + (max_y-min_y)/2.0f;
    b.w = max_x - min_x;
    b.h = max_y - min_x;
    return b;
}

static void cost_and_delta(face_aliment_layer l,network net)
{
    float center_x=0,center_y=0;
    int index_x=0,index_y=0;
    int truth_start_index=0;
    int batch_index = 0;
    int i=0,j=0;
    int b=0;

    for(b=0;b<l.batch;++b){
        for(i=0;i<68;++i){
            center_x += net.truth[i*2+b*l.c];
            center_y += net.truth[i*2+1+b*l.c];
        }
        center_x /= 68;
        center_y /= 68;
        index_x = (int)(center_x*l.w);
        index_y = (int)(center_y*l.h);
        float *truth = net.truth+b*136;
        for(i=0;i<137;++i){
            float *data = l.output+i*l.w*l.h+b*l.outputs;
            float *diff = l.delta+i*l.w*l.h+b*l.outputs;
            //printf("i is %d,%d,%d,%d,%d\n",i,l.w,l.h,b,l.outputs);
            for(j=0;j<l.w*l.h;++j){
                if((index_y*l.w+index_x)==j){
                    if(i==0){
                        diff[j] = 1-data[j];
                        l.cost[0] += 0.5f*diff[j]*diff[j];
                    }else{
                        if(i%2==0){
                            diff[j] = truth[i-1]*l.h-data[j];
                            l.cost[0] += 0.5f*diff[j]*diff[j];
                        }else{
                            diff[j] = truth[i-1]*l.w-data[j];
                            l.cost[0] += 0.5f*diff[j]*diff[j];
                        }
                    }
                }else{
                    diff[j] = 0-data[j];
                    l.cost[0] += 0.5f*diff[j]*diff[j];
                }
            }
        }
    }
    
    l.cost[0] /= l.batch;
}

void forward_face_aliment_layer(face_aliment_layer l, network net)
{
    memcpy(l.output, net.input, l.inputs*l.batch*sizeof(float));
    activate_array(l.output, l.w*l.h*3, LOGISTIC);
    activate_array(l.output+l.w*l.h*5, l.w*l.h*136, TANH);
}

void backward_face_aliment_layer(face_aliment_layer l, network net)
{
    cost_and_delta(l,net);
    gradient_array(l.output, l.w*l.h*3, LOGISTIC, l.delta);
    gradient_array(l.output, l.w*l.h*136, TANH, l.delta);
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}


#ifdef GPU
void forward_face_aliment_layer_gpu(face_aliment_layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.w*l.h, LOGISTIC);
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
}

void backward_face_aliment_layer_gpu(face_aliment_layer l, network net)
{
    cost_and_delta(l,net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    gradient_array_gpu(l.output_gpu, l.w*l.h, LOGISTIC, l.delta_gpu);
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
