#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_face_detect_layer(int batch, int w, int h, int n, int classes, int coords);
void forward_face_detect_layer(const layer l, network net);
void backward_face_detect_layer(const layer l, network net);
void resize_face_detect_layer(layer *l, int w, int h);

#ifdef GPU
void forward_face_detect_layer_gpu(const layer l, network net);
void backward_face_detect_layer_gpu(layer l, network net);
#endif

#endif
