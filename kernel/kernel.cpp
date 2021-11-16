#include <cassert>
#include <iostream>
#include <limits>

#include "kernel.h"

#include <omp.h>

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;


void _gspmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{
    //cout << "spmm " << op << "reverse = " << reverse << endl;
   //cout<<"value of snaph: " << snaph <<endl;
   //cout<<"whats in output "<<output;
   //cout<<"Whats in input"<<input;
    //op.h has class definition for input and output-> both array2d_t row and cols are in64_t(public variables)
    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here.    
    
    int64_t col_count = output.col_count;
    vid_t* nebrs = snaph->nebrs;
    vid_t* offset = snaph->offset;
    vid_t vertex_count = snaph->get_vcount();


   
    //cout<<"what is vertex_count: "<<vertex_count<<endl;

    
 #pragma omp parallel   
{
	#pragma omp for
    for(int i = 0; i < vertex_count; i++) {
        vid_t deg = snaph->get_degree(i);
        if (!reverse){  

            input.row_normalize(i, deg + 1);
            output.row_add(input.data_ptr + i * col_count, i);
            for(int j = offset[i]; j < offset[i+1]; j++) {
                output.row_add(input.data_ptr + nebrs[j] * col_count, nebrs[j]);
            }


        } else {
            output.row_add(input.data_ptr + i * col_count, i);
            output.row_normalize(i, deg + 1);
        }
    }  
}         


}


void invoke_gspmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
         return _gspmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
         return _gspmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }
}
