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
    l.truths = 137;
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

void cost_and_delta(face_aliment_layer l,network net)
{
    float center_x=0,center_y=0;
    int index_x=0,index_y=0;
    int truth_start_index=0;
    int batch_index = 0;
    int i=0;
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
        //printf("%d,%d\n",index_x,index_y);
        //cost
        truth_start_index = (index_y*l.w+index_x)*l.c+b*l.c;
        batch_index = b*l.inputs;
        for(i=0;i<truth_start_index;++i){
            l.cost[0] += 0.5f*l.output[i+batch_index]*l.output[i+batch_index];
            l.delta[i+batch_index] = 0-l.output[i+batch_index];
        }
        for(i=truth_start_index;i<truth_start_index+l.c;++i){
            if(i==truth_start_index){
                l.delta[i+batch_index] = 1-l.output[i+batch_index];
                l.cost[0] += 0.5f*(l.output[i+batch_index]-1)*(l.output[i+batch_index]-1);
            }else{
                l.cost[0] += 0.5f*(l.output[i+batch_index]-net.truth[i-truth_start_index-1+b*l.c])*(l.output[i+batch_index]-net.truth[i-truth_start_index-1+b*l.c]);
                l.delta[i+batch_index] = net.truth[i-truth_start_index-1+b*l.c]-l.output[i+batch_index];
            }
        }
        for(i=truth_start_index+l.c;i<l.inputs;++i){
            l.cost[0] += 0.5f*l.output[i+batch_index]*l.output[i+batch_index];
            l.delta[i+batch_index] = 0-l.output[i+batch_index];
        }
    }
    
    l.cost[0] /= l.batch;
}

void forward_face_aliment_layer(face_aliment_layer l, network net)
{
    memcpy(l.output, net.input, l.inputs*l.batch*sizeof(float));
    activate_array(l.output, l.inputs*l.batch, LOGISTIC);
}

void backward_face_aliment_layer(face_aliment_layer l, network net)
{
    cost_and_delta(l,net);
    gradient_array(l.output, l.inputs*l.batch, LOGISTIC, l.delta);
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}


#ifdef GPU
void forward_face_aliment_layer_gpu(face_aliment_layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.inputs*l.batch, LOGISTIC);
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
}

void backward_face_aliment_layer_gpu(face_aliment_layer l, network net)
{
    cost_and_delta(l,net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    gradient_array_gpu(l.output_gpu, l.inputs*l.batch, LOGISTIC, l.delta_gpu);
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
