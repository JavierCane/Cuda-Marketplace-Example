#include <iostream>
#include <vector>
#include <algorithm> // for_each function

using namespace std;

#define NUM_PRODUCTS 2
#define NUM_BUY_OPTIONS 2

void printAllProductsAllBuyOptions(int *all_products_buy_options);

void getBestBuyOptions(int *all_products_buy_options, int *best_buy_options);

void printBestBuyOptions(int *best_buy_options);

int main()
{
    // START Vector with all the buy options indexed by product
    int *all_products_buy_options;
    int *best_buy_options;
    all_products_buy_options = (int *) malloc(NUM_PRODUCTS*NUM_BUY_OPTIONS*2*sizeof(int));
    best_buy_options = (int *) malloc(NUM_PRODUCTS*2*sizeof(int));

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

    // END Vector with all the buy options indexed by product

    printAllProductsAllBuyOptions(all_products_buy_options);

    getBestBuyOptions(all_products_buy_options, best_buy_options);
    printBestBuyOptions(best_buy_options);
    return 0;
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
    for(int i = 0; i < NUM_PRODUCTS; ++i){
        int best_store = all_products_buy_options[i*NUM_BUY_OPTIONS*2];
        int best_price = all_products_buy_options[i*NUM_BUY_OPTIONS*2+1];
        for(int j = 2; j < NUM_BUY_OPTIONS*2; j += 2){
            if (all_products_buy_options[i*NUM_BUY_OPTIONS*2+j+1] < best_price){
               best_store = all_products_buy_options[i*NUM_BUY_OPTIONS*2+j];
               best_price = all_products_buy_options[i*NUM_BUY_OPTIONS*2+j+1];
            }
        }
        best_buy_options[i*2] = best_store;
        best_buy_options[i*2+1] = best_price;
    }
}

void printBestBuyOptions(int *best_buy_options)
{
    cout << endl <<"Best products buy options:" << endl;
    for (int i = 0; i < NUM_PRODUCTS*2; i+=2){
        cout << endl << "\tproduct_id: " << i/2 << endl;
        cout << "Buy option:" << endl;
        cout << "\tstore_id: " << best_buy_options[i] << endl;
        cout << "\tprice: " << best_buy_options[i+1] << endl;
    }
}
