#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define DEBUG_LEVEL 0

#define ELEMENTS_PER_BUY_OPTION 2
#define STORE_ID_OFFSET 0
#define PRICE_OFFSET 1

#define NUM_PRODUCTS 30000
#define NUM_BUY_OPTIONS 1024 // Debe ser igual al número de threads.

#define NUM_THREADS 1024 // El número mínimo de threads es 32 (por el tamaño de warp) y el maximo 1024.

void initAllProductsBuyOptions(unsigned int *all_products_buy_options);

void printAllProductsAllBuyOptions(unsigned int *all_products_buy_options);
void getBestBuyOptions(unsigned int *all_products_buy_options, unsigned int *best_buy_options);
void printBestBuyOptions(unsigned int *best_buy_options);

bool areResultsValid(unsigned int *all_products_buy_options, unsigned int *best_buy_options);

// ToDo: Cada thread ejecuta el kernel.
// Identificar el thread en el que estamos, en base a esto, calculamos si nos toca trabajar y, en caso afirmativo
// comparar entre 2 buy options cuál es la mejor y dejarla en el vector de memoria compartida (temporal)
// Hacemos __syncthreads() para que no haya colisiones y que todos hayan acabado esta ronda de comparación. Pasar a iteración siguiente.
// Cuando se haya agotado el bloque, pasar las opciones del vector temporal al vector de salida en caso de ser el thread 0.

__global__ void KernelKnapsack(unsigned int *total_buy_options, unsigned int *best_buy_options)
{
    __shared__ unsigned int tmp_best_buy_options[NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION];

    unsigned int thread_id = threadIdx.x;
    unsigned int shared_thread_buy_option = thread_id * ELEMENTS_PER_BUY_OPTION;
    unsigned int global_thread_buy_option = ( blockIdx.x * blockDim.x + thread_id ) * ELEMENTS_PER_BUY_OPTION;

    tmp_best_buy_options[shared_thread_buy_option + STORE_ID_OFFSET] = total_buy_options[global_thread_buy_option + STORE_ID_OFFSET];
    tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] = total_buy_options[global_thread_buy_option + PRICE_OFFSET];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {
            unsigned int next_buy_option_position = shared_thread_buy_option + stride * ELEMENTS_PER_BUY_OPTION;

            if (tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] > tmp_best_buy_options[next_buy_option_position + PRICE_OFFSET])
            {
                tmp_best_buy_options[shared_thread_buy_option + STORE_ID_OFFSET] = tmp_best_buy_options[next_buy_option_position + STORE_ID_OFFSET];
                tmp_best_buy_options[shared_thread_buy_option + PRICE_OFFSET] = tmp_best_buy_options[next_buy_option_position + PRICE_OFFSET];
            }
        }
        __syncthreads();
    }

    if (thread_id == 0)
    {
        best_buy_options[blockIdx.x * ELEMENTS_PER_BUY_OPTION + STORE_ID_OFFSET] = tmp_best_buy_options[STORE_ID_OFFSET];
        best_buy_options[blockIdx.x * ELEMENTS_PER_BUY_OPTION + PRICE_OFFSET] = tmp_best_buy_options[PRICE_OFFSET];
    }
}

int main(int argc, char** argv)
{
    // Buy options in host and device
    unsigned int *host_all_products_buy_options = (unsigned int *) malloc( NUM_PRODUCTS * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION * sizeof(unsigned int) );
    unsigned int *best_buy_options = (unsigned int *) malloc( NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION * sizeof(unsigned int) );
    unsigned int *device_all_products_buy_options;
    unsigned int *device_best_buy_options;

    // Metadata
    unsigned int buy_option_size = ELEMENTS_PER_BUY_OPTION * sizeof(unsigned int);
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
    cudaMalloc( (unsigned int**) &device_all_products_buy_options, total_buy_options_size );
    cudaMalloc( (unsigned int**) &device_best_buy_options, best_buy_options_size );

    // Copiar datos desde el host en el device
    cudaMemcpy(device_all_products_buy_options, host_all_products_buy_options, total_buy_options_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    // Ejecutar el kernel (número de bloques = número de productos). Un bloque por cada producto y todos los threads por cada bloque.
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
    printf("Number of products: %d\n", NUM_PRODUCTS);
    printf("Number of buy options per product: %d\n", NUM_BUY_OPTIONS);
    printf("Vector Size: %d\n", num_total_buy_options);
    printf("Number of Threads: %d\n", NUM_THREADS);
    printf("Number of blocks (products): %d\n", NUM_PRODUCTS);
    printf("Total time %4.6f milseg\n", elapsed_time);
    printf("Bandwidth %4.3f GB/s\n", (num_total_buy_options * sizeof(unsigned int)) / (1000000 * elapsed_time));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (DEBUG_LEVEL >= 2)
    {
        printAllProductsAllBuyOptions(host_all_products_buy_options);
    }

    if (DEBUG_LEVEL >= 1)
    {
        printBestBuyOptions(best_buy_options);
    }

    if ( areResultsValid( host_all_products_buy_options, best_buy_options ) )
    {
        printf ("TEST PASS\n");
    }
    else
    {
        printf ("TEST FAIL\n");
    }
}

void initAllProductsBuyOptions(unsigned int *all_products_buy_options)
{
    for(unsigned int product_iteration = 0; product_iteration < NUM_PRODUCTS; ++product_iteration)
    {
        unsigned int current_product_position = product_iteration * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION;

        for(unsigned int buy_option_iteration = 0; buy_option_iteration < NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION; buy_option_iteration += ELEMENTS_PER_BUY_OPTION)
        {
            unsigned int current_product_store_position = current_product_position + buy_option_iteration + STORE_ID_OFFSET;
            unsigned int current_product_price_position = current_product_position + buy_option_iteration + PRICE_OFFSET;

            // Set the current product buy option to the store with the same id as the current iteration in order to do not duplicate buy options.
            all_products_buy_options[current_product_store_position] = buy_option_iteration/2;

            // Set the price with a random value between 1 and 1000.
            all_products_buy_options[current_product_price_position] = rand() % 999 + 1;
        }
    }
}

bool areResultsValid(unsigned int *all_products_buy_options, unsigned int *best_buy_options_by_device)
{
   unsigned int *best_buy_options_by_host = (unsigned int *) malloc( NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION * sizeof(unsigned int) );

   getBestBuyOptions(all_products_buy_options, best_buy_options_by_host);

   if (DEBUG_LEVEL >= 1)
   {
       printBestBuyOptions(best_buy_options_by_host);
   }

   for (unsigned int product_iteration = 0; product_iteration < NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION; product_iteration += ELEMENTS_PER_BUY_OPTION)
   {
       unsigned int current_product_store_position = product_iteration + STORE_ID_OFFSET;
       unsigned int current_product_price_position = product_iteration + PRICE_OFFSET;

       unsigned int best_store_by_device = best_buy_options_by_device[current_product_store_position];
       unsigned int best_price_by_device = best_buy_options_by_device[current_product_price_position];

       unsigned int best_store_by_host = best_buy_options_by_host[current_product_store_position];
       unsigned int best_price_by_host = best_buy_options_by_host[current_product_price_position];

       if (best_store_by_device != best_store_by_host || best_price_by_device != best_price_by_host)
       {
           if (DEBUG_LEVEL >= 1)
           {
               printf("FAILED IN product: %d\n", product_iteration);
               printf("\tbest_store_by_device: %d\n", best_store_by_device);
               printf("\tbest_store_by_host: %d\n", best_store_by_host);
               printf("\tbest_price_by_device: %d\n", best_price_by_device);
               printf("\tbest_price_by_host: %d\n", best_price_by_host);
           }

           return false;
       }
   }

   return true;
}

void printAllProductsAllBuyOptions(unsigned int *all_products_buy_options)
{
    cout << "All products buy options:" << endl;
    for (unsigned int i = 0; i < NUM_PRODUCTS; ++i){
        cout << endl << "\tproduct_id: " << i << endl;
        for (unsigned int j = 0; j < NUM_BUY_OPTIONS*2; j += 2){
            cout << "Buy option:" << endl;
            cout << "\tstore_id: " << all_products_buy_options[i*NUM_BUY_OPTIONS*2+j] << endl;
            cout << "\tprice: " << all_products_buy_options[i*NUM_BUY_OPTIONS*2+j+1] << endl;
        }
    }
}

void getBestBuyOptions(unsigned int *all_products_buy_options, unsigned int *best_buy_options)
{
    for(unsigned int product_iteration = 0; product_iteration < NUM_PRODUCTS; ++product_iteration)
    {
        unsigned int current_product_position = product_iteration * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION;
        unsigned int current_product_store_position = current_product_position + STORE_ID_OFFSET;
        unsigned int current_product_price_position = current_product_position + PRICE_OFFSET;

        unsigned int best_store = all_products_buy_options[current_product_store_position];
        unsigned int best_price = all_products_buy_options[current_product_price_position];

        for(unsigned int product_to_compare = ELEMENTS_PER_BUY_OPTION; product_to_compare < NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION; product_to_compare += ELEMENTS_PER_BUY_OPTION)
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

void printBestBuyOptions(unsigned int *best_buy_options)
{
    cout << endl <<"Best products buy options:" << endl;
    for (unsigned int i = 0; i < NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION; i += ELEMENTS_PER_BUY_OPTION)
    {
        cout << "Best buy option for product_id: " << i / ELEMENTS_PER_BUY_OPTION << endl;
        cout << "\tstore_id: " << best_buy_options[i + STORE_ID_OFFSET] << endl;
        cout << "\tprice: " << best_buy_options[i + PRICE_OFFSET] << endl;
    }
}
