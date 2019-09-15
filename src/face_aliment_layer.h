#ifndef FACE_ALIMENT_LAYER_H
#define FACE_ALIMENT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer face_aliment_layer;

face_aliment_layer make_face_aliment_layer(int batch, int w, int h, int n);

void forward_face_aliment_layer(face_aliment_layer l, network net);
void backward_face_aliment_layer(face_aliment_layer l, network net);
void resize_face_aliment_layer(face_aliment_layer *l, int w, int h);

#ifdef GPU
void forward_face_aliment_layer_gpu(face_aliment_layer l, network net);
void backward_face_aliment_layer_gpu(face_aliment_layer l, network net);
#endif

#endif
