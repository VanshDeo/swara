#ifndef SWARA_MODEL_DATA_H_
#define SWARA_MODEL_DATA_H_

#include <cstdint>

// Swara Keyword Spotting / Edge Wake Word Detection Model
// Generated from: swara_int8.tflite (25904 bytes)
// Model Metadata:
//   Input shape:         [1, 49, 10, 1]
//   Input dtype:         <class 'numpy.int8'>
//   Input scale:         1.3123881816864014 (zero-point: 83)
//   Output shape:        [1, 3]
//   Output dtype:        <class 'numpy.int8'>
//   Output scale:        0.00390625 (zero-point: -128)
//   Total weights count: 11317
//   Operators (8):   CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX
//   TFLM compatible:     True
extern const unsigned char g_swara_model_data[];
extern const unsigned int g_swara_model_data_len;

#endif  // SWARA_MODEL_DATA_H_
