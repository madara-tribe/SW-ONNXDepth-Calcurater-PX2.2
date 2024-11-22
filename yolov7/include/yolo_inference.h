#include <iostream>

typedef struct Result {
    int x1;
    int x2;
    int y1;
    int y2;
    int obj_id;
    float accuracy;

    Result(int x1_, int x2_, int y1_, int y2_, int obj_id_, float accuracy_) {
       x1 = x1_;
       x2 = x2_;
       y1 = y1_;
       y2 = y2_;
       obj_id = obj_id_;
       accuracy = accuracy_;
   }

} result_t ;

int model_input_width;
int model_input_height;
int pad_size_y;
int pad_size_x;
int model_width_after_padding;
int model_height_after_padding;
