#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define PRODUCT_ID_POS 0
#define STORE_ID_POS 1
#define PRICE_POS 2

#define NUM_PRODUCTS 3
#define NUM_BUY_OPTIONS 6144
#define ELEMENTS_PER_BUY_OPTION 2

#define NUM_THREADS 512 // El número mínimo de threads es 32 (por el tamaño de warp) y el maximo 1024

void printAllProductsAllBuyOptions(vector< vector< vector< int > > > all_products_buy_options);
void printProductAllBuyOption(vector< vector< int > > product_buy_options);
void printProductBuyOption(vector< int > buy_option);

void getBestBuyOptions(vector< vector< vector< int > > > all_products_buy_options, vector< vector< int > > *best_buy_options);
void getProductBestBuyOption(vector< vector< int > > product_buy_options, vector< int > *best_buy_option);
void printBestBuyOption(vector< vector< int > > best_buy_options);

void initTotalBuyOptions(int num_total_buy_options, int *all_products_buy_options);
int checkResults(int num_total_buy_options, int *all_products_buy_options, int *host_best_buy_options);

// ToDo: Cada thread ejecuta el kernel.
// Identificar el thread en el que estamos, en base a esto, calculamos si nos toca trabajar y, en caso afirmativo
// comparar entre 2 buy options cuál es la mejor y dejarla en el vector de memoria compartida (temporal)
// Hacemos __syncthreads() para que no haya colisiones y que todos hayan acabado esta ronda de comparación. Pasar a iteración siguiente.
// Cuando se haya agotado el bloque, pasar las opciones del vector temporal al vector de salida en caso de ser el thread 0.
__global__ void KernelKnapsack(unsigned int *total_buy_options, unsigned int *best_buy_options, unsigned int total_buy_options_size)
{
    __shared__ int tmp_best_buy_options[total_buy_options_size];
    unsigned int stride;

    // Cada thread carga 1 elemento desde la memoria global
    unsigned int thread_id = threadIdx.x;
    unsigned int thread_product = blockIdx.x * blockDim.x + threadIdx.x;
    tmp_best_buy_options[thread_id] = total_buy_options[thread_product];
    __syncthreads();

    // Hacemos la reduccion en la memoria compartida
    for(stride = 1; stride < blockDim.x; stride *= 4) {
      if (thread_id % (4 * stride) == 0) {
          // ToDo: Revisar comparación en función de la nueva definición de la estructura del array.
          if (tmp_best_buy_options[thread_id + stride][1] < tmp_best_buy_options[thread_id][1]) {
              tmp_best_buy_options[thread_id] += tmp_best_buy_options[thread_id + stride];
          }
      }
      __syncthreads();
    }

    // El thread 0 escribe el resultado de este bloque en la memoria global
    if (thread_id == 0) {
        best_buy_options[blockIdx.x] = tmp_best_buy_options[0];
    }
}

int main(int argc, char** argv)
{
    // START Vector with all the buy options indexed by product
    vector< vector< vector< int > > > all_products_buy_options;

    // START Vector all the buy options for the product 1
    vector< vector<int> > product_1_buy_options;

    vector<int> product_1_store_1_buy_option;
    product_1_store_1_buy_option.push_back(1); // product_id
    product_1_store_1_buy_option.push_back(1); // store_id
    product_1_store_1_buy_option.push_back(11); // price

    vector<int> product_1_store_2_buy_option;
    product_1_store_2_buy_option.push_back(1); // product_id
    product_1_store_2_buy_option.push_back(2); // store_id
    product_1_store_2_buy_option.push_back(12); // price

    vector<int> product_1_store_3_buy_option;
    product_1_store_3_buy_option.push_back(1); // product_id
    product_1_store_3_buy_option.push_back(3); // store_id
    product_1_store_3_buy_option.push_back(9); // price

    product_1_buy_options.push_back(product_1_store_1_buy_option);
    product_1_buy_options.push_back(product_1_store_2_buy_option);
    product_1_buy_options.push_back(product_1_store_3_buy_option);
    // END Vector all the buy options for the product 1

    // START Vector all the buy options for the product 2
    vector< vector<int> > product_2_buy_options;

    vector<int> product_2_store_1_buy_option;
    product_2_store_1_buy_option.push_back(2); // product_id
    product_2_store_1_buy_option.push_back(1); // store_id
    product_2_store_1_buy_option.push_back(21); // price

    product_2_buy_options.push_back(product_2_store_1_buy_option);
    // END Vector all the buy options for the product 2

    all_products_buy_options.push_back(product_1_buy_options);
    all_products_buy_options.push_back(product_2_buy_options);
    // END Vector with all the buy options indexed by product

    // START Cuda init
    unsigned int num_total_buy_options;
    unsigned int total_buy_options_size, best_buy_options_size;
    unsigned int NUM_PRODUCTS, nThreads;

    float elapsedTime;
    cudaEvent_t start, stop;

    unsigned int *host_total_buy_options, *host_best_buy_options;
    unsigned int *device_total_buy_options, *device_best_buy_options;
    unsigned int buy_option_size = ELEMENTS_PER_BUY_OPTION * sizeof(int);

    num_total_buy_options = NUM_PRODUCTS * NUM_BUY_OPTIONS;

    total_buy_options_size = num_total_buy_options * buy_option_size;
    best_buy_options_size = NUM_PRODUCTS * buy_option_size;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Obtener Memoria en el host
    host_total_buy_options = (int*) malloc(total_buy_options_size);
    host_best_buy_options = (int*) malloc(best_buy_options_size);

    // Obtiene Memoria [pinned] en el host
    //cudaMallocHost((float**)&h_v, numBytesV);
    //cudaMallocHost((float**)&h_w, numBytesW);

    // Inicializa los vectores
    InitV(num_total_buy_options, host_total_buy_options);

    // Obtener Memoria en el device
    cudaMalloc((float**)&device_total_buy_options, total_buy_options_size);
    cudaMalloc((float**)&device_best_buy_options, best_buy_options_size);

    // Copiar datos desde el host en el device
    cudaMemcpy(device_total_buy_options, host_total_buy_options, total_buy_options_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    // Ejecutar el kernel (número de bloques = número de productos)
    KernelKnapsack<<<NUM_PRODUCTS, NUM_THREADS>>>(device_total_buy_options, device_best_buy_options, total_buy_options_size);

    // Obtener el resultado parcialdesde el host
    cudaMemcpy(host_best_buy_options, device_best_buy_options, best_buy_options_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Liberar Memoria del device
    cudaFree(device_total_buy_options);
    cudaFree(device_best_buy_options);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nKERNEL KNAPSACK\n");
    printf("Vector Size: %d\n", num_total_buy_options);
    printf("Number of Threads: %d\n", NUM_THREADS);
    printf("Number of blocks (products): %d\n", NUM_PRODUCTS);
    printf("Total time %4.6f milseg\n", elapsedTime);
    printf("Bandwidth %4.3f GB/s\n", (num_total_buy_options * sizeof(int)) / (1000000 * elapsedTime));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ToDo: Ejecutar checkResults();

    return 0;
}


// ToDo: Mover aquí la inicialización de all_products_buy_options
void initTotalBuyOptions(int num_total_buy_options, int *all_products_buy_options)
{
}

// ToDo: llamar a getBestBuyOptions y comprobar que sea igual que el host_best_buy_options
int checkResults(int num_total_buy_options, int *all_products_buy_options, int *host_best_buy_options)
{
   return 1;
}

// ToDo: calcular mejor opción desde el host con la nueva estructura de arrays y no vectores
void getBestBuyOptions(vector< vector< vector< int > > > all_products_buy_options, vector< vector< int > > *best_buy_options)
{
    int num_products = all_products_buy_options.size();

    for(int current_product_iteration = 0; current_product_iteration < num_products; ++current_product_iteration) {

        // Initialize the tmp_best_buy_option with the first buy option
        vector<int> tmp_best_buy_option = all_products_buy_options[current_product_iteration][0];

        int num_product_buy_options = all_products_buy_options[current_product_iteration].size();

        for(int buy_option_iteration = 1; buy_option_iteration < num_product_buy_options; ++buy_option_iteration) {

            int current_product_price = all_products_buy_options[current_product_iteration][buy_option_iteration][PRICE_POS];

            if (current_product_price < tmp_best_buy_option[PRICE_POS]) {
                tmp_best_buy_option = all_products_buy_options[current_product_iteration][buy_option_iteration];
            }
        }

        (*best_buy_options).push_back(tmp_best_buy_option);
    }
}

