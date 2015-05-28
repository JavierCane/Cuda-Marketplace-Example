#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define ELEMENTS_PER_BUY_OPTION 2
#define STORE_ID_OFFSET 0
#define PRICE_OFFSET 1

#define NUM_PRODUCTS 2
#define NUM_BUY_OPTIONS 2

#define NUM_THREADS 512 // El número mínimo de threads es 32 (por el tamaño de warp) y el maximo 1024

void initAllProductsBuyOptions(int *all_products_buy_options);

void printAllProductsAllBuyOptions(int *all_products_buy_options);
void getBestBuyOptions(int *all_products_buy_options, int *best_buy_options);
void printBestBuyOptions(int *best_buy_options);

int main(int argc, char** argv)
{
    // Buy options in host and device
    int *host_all_products_buy_options = (int *) malloc( NUM_PRODUCTS * NUM_BUY_OPTIONS * ELEMENTS_PER_BUY_OPTION * sizeof(int) );
    int *best_buy_options = (int *) malloc( NUM_PRODUCTS * ELEMENTS_PER_BUY_OPTION * sizeof(int) );

    initAllProductsBuyOptions(host_all_products_buy_options);

    // DEBUG
    printAllProductsAllBuyOptions(host_all_products_buy_options);
    getBestBuyOptions(host_all_products_buy_options, best_buy_options);
    printBestBuyOptions(best_buy_options);
    // END DEBUG
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
