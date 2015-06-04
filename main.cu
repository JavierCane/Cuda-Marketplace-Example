#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define ELEMENTS_PER_BUY_OPTION 2
#define STORE_ID_OFFSET 0
#define PRICE_OFFSET 1

#define NUM_PRODUCTS 3
#define NUM_BUY_OPTIONS 6144

#define NUM_THREADS 512 // El número mínimo de threads es 32 (por el tamaño de warp) y el maximo 1024

void initAllProductsBuyOptions(int *all_products_buy_options);

void printAllProductsAllBuyOptions(int *all_products_buy_options);
void getBestBuyOptions(int *all_products_buy_options, int *best_buy_options);
void printBestBuyOptions(int *best_buy_options);

bool areResultsValid(int *all_products_buy_options, int *best_buy_options);

// ToDo: Cada thread ejecuta el kernel.
// Identificar el thread en el que estamos, en base a esto, calculamos si nos toca trabajar y, en caso afirmativo
// comparar entre 2 buy options cuál es la mejor y dejarla en el vector de memoria compartida (temporal)
// Hacemos __syncthreads() para que no haya colisiones y que todos hayan acabado esta ronda de comparación. Pasar a iteración siguiente.
// Cuando se haya agotado el bloque, pasar las opciones del vector temporal al vector de salida en caso de ser el thread 0.

__global__ void KernelKnapsack(unsigned int *total_buy_options, unsigned int *best_buy_options)
{
    __shared__ float tmp_best_buy_options[1024];
    unsigned int stride;

    unsigned int thread_id = threadIdx.x;
    unsigned int thread_product = blockIdx.x * blockDim.x + threadIdx.x * ELEMENTS_PER_BUY_OPTION;
    tmp_best_buy_options[thread_id + STORE_ID_OFFSET] = total_buy_options[thread_product + STORE_ID_OFFSET];
    tmp_best_buy_options[thread_id + PRICE_OFFSET] = total_buy_options[thread_product + PRICE_OFFSET];
    __syncthreads();

    for (stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadid % (2 * stride) == 0)
        {
            unsigned int next_buy_option_position = thread_id + stride * ELEMENTS_PER_BUY_OPTION;

            if (tmp_best_buy_options[thread_id + PRICE_OFFSET] > tmp_best_buy_options[next_buy_option_position + PRICE_OFFSET])
            {
                tmp_best_buy_options[thread_id + STORE_ID_OFFSET] = tmp_best_buy_options[next_buy_option_position + STORE_ID_OFFSET];
                tmp_best_buy_options[thread_id + PRICE_OFFSET] = tmp_best_buy_options[next_buy_option_position + PRICE_OFFSET];
            }
        }
        __syncthreads();
    }

    if (thread_id == 0)
    {
        best_buy_options[blockIdx.x + STORE_ID_OFFSET] = tmp_best_buy_options[STORE_ID_OFFSET];
        best_buy_options[blockIdx.x + PRICE_OFFSET] = tmp_best_buy_options[PRICE_OFFSET];
    }
}


int main(int argc, char** argv)
{
    // Buy options in host and device
    int *host_all_products_buy_options = (int *) malloc( NUM_PRODUCTS * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION * sizeof(int) );
    int *best_buy_options = (int *) malloc( NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION * sizeof(int) );
    unsigned int *device_all_products_buy_options;
    unsigned int *device_best_buy_options;

    // Metadata
    unsigned int buy_option_size = ELEMENTS_PER_BUY_OPTION * sizeof(int);
    unsigned int num_total_buy_options = NUM_PRODUCTS * NUM_BUY_OPTIONS;
    unsigned int total_buy_options_size = num_total_buy_options * buy_option_size;
    unsigned int best_buy_options_size = NUM_PRODUCTS * buy_option_size;

    // Benchmarking
    float elapsed_time;
    cudaEvent_t start;
    cudaEvent_t stop;

    initAllProductsBuyOptions(host_all_products_buy_options);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Obtener Memoria en el device
    cudaMalloc( (float**) &device_all_products_buy_options, total_buy_options_size );
    cudaMalloc( (float**) &device_best_buy_options, best_buy_options_size );

    // Copiar datos desde el host en el device
    cudaMemcpy(device_all_products_buy_options, host_all_products_buy_options, total_buy_options_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    // Ejecutar el kernel (número de bloques = número de productos)
    KernelKnapsack<<<NUM_PRODUCTS, NUM_THREADS>>>(device_all_products_buy_options, device_best_buy_options);

    // Obtener el resultado parcial desde el host
    cudaMemcpy(best_buy_options, device_best_buy_options, best_buy_options_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Liberar Memoria del device
    cudaFree(device_all_products_buy_options);
    cudaFree(device_best_buy_options);

    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("\nKERNEL KNAPSACK\n");
    printf("Vector Size: %d\n", num_total_buy_options);
    printf("Number of Threads: %d\n", NUM_THREADS);
    printf("Number of blocks (products): %d\n", NUM_PRODUCTS);
    printf("Total time %4.6f milseg\n", elapsed_time);
    printf("Bandwidth %4.3f GB/s\n", (num_total_buy_options * sizeof(int)) / (1000000 * elapsed_time));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // DEBUG
    printAllProductsAllBuyOptions(host_all_products_buy_options);
    getBestBuyOptions(host_all_products_buy_options, best_buy_options);
    printBestBuyOptions(best_buy_options);
    // END DEBUG

    if ( areResultsValid( host_all_products_buy_options, best_buy_options ) )
    {
        printf ("TEST PASS\n");
    }
    else
    {
        printf ("TEST FAIL\n");
    }
}

void initAllProductsBuyOptions(int *all_products_buy_options)
{
    // START Vector all the buy options for the product 1

    // product_1_store_1_buy_option;
    all_products_buy_options[0] = 11; // store_id
    all_products_buy_options[1] = 11;  // price

    // product_1_store_2_buy_option;
    all_products_buy_options[2] = 12; // store_id
    all_products_buy_options[3] = 12;  // price

    // END Vector all the buy options for the product 1

    // START Vector all the buy options for the product 2

    // product_1_store_1_buy_option;
    all_products_buy_options[4] = 21; // store_id
    all_products_buy_options[5] = 21;  // price

    // product_1_store_2_buy_option;
    all_products_buy_options[6] = 22; // store_id
    all_products_buy_options[7] = 2;  // price

    // END Vector all the buy options for the product 2
}

bool areResultsValid(int *all_products_buy_options, int *best_buy_options)
{
   int *tmp_best_buy_options;

   getBestBuyOptions(all_products_buy_options, tmp_best_buy_options);

   for (int product_iteration = 0; product_iteration < NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION; product_iteration += ELEMENTS_PER_BUY_OPTION)
   {
       int current_product_position = product_iteration * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION;
       int current_product_store_position = current_product_position + STORE_ID_OFFSET;
       int current_product_price_position = current_product_position + PRICE_OFFSET;

       int best_store = best_buy_options[current_product_store_position];
       int best_price = best_buy_options[current_product_price_position];

       int tmp_best_store = tmp_best_buy_options[current_product_store_position];
       int tmp_best_price = tmp_best_buy_options[current_product_price_position];

       if (best_store != tmp_best_store || best_price != tmp_best_price)
       {
           return false;
       }
   }

   return true;
}

void printAllProductsAllBuyOptions(int *all_products_buy_options)
{
    cout << "All products buy options:" << endl;
    for (int i = 0; i < NUM_PRODUCTS; ++i){
        cout << endl << "\tproduct_id: " << i << endl;
        for (int j = 0; j < NUM_BUY_OPTIONS*2; j += 2){
            cout << "Buy option:" << endl;
            cout << "\tstore_id: " << all_products_buy_options[i*NUM_BUY_OPTIONS*2+j] << endl;
            cout << "\tprice: " << all_products_buy_options[i*NUM_BUY_OPTIONS*2+j+1] << endl;
        }
    }
}

void getBestBuyOptions(int *all_products_buy_options, int *best_buy_options)
{
    for(int product_iteration = 0; product_iteration < NUM_PRODUCTS; ++product_iteration)
    {
        int current_product_position = product_iteration * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION;
        int current_product_store_position = current_product_position + STORE_ID_OFFSET;
        int current_product_price_position = current_product_position + PRICE_OFFSET;

        int best_store = all_products_buy_options[current_product_store_position];
        int best_price = all_products_buy_options[current_product_price_position];

        for(int product_to_compare = ELEMENTS_PER_BUY_OPTION; product_to_compare < NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION; product_to_compare += ELEMENTS_PER_BUY_OPTION)
        {
            if (all_products_buy_options[current_product_position + product_to_compare + PRICE_OFFSET] < best_price)
            {
               best_store = all_products_buy_options[product_iteration * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION + product_to_compare + STORE_ID_OFFSET];
               best_price = all_products_buy_options[product_iteration * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION + product_to_compare + PRICE_OFFSET];
            }
        }
        best_buy_options[product_iteration * ELEMENTS_PER_BUY_OPTION + STORE_ID_OFFSET] = best_store;
        best_buy_options[product_iteration * ELEMENTS_PER_BUY_OPTION + PRICE_OFFSET] = best_price;
    }
}

void printBestBuyOptions(int *best_buy_options)
{
    cout << endl <<"Best products buy options:" << endl;
    for (int i = 0; i < NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION; i += ELEMENTS_PER_BUY_OPTION)
    {
        cout << "Best buy option for product_id: " << i / ELEMENTS_PER_BUY_OPTION << endl;
        cout << "\tstore_id: " << best_buy_options[i + STORE_ID_OFFSET] << endl;
        cout << "\tprice: " << best_buy_options[i + PRICE_OFFSET] << endl;
    }
}
