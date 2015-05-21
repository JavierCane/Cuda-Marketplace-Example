#include <iostream>
#include <vector>
#include <algorithm> // for_each function

using namespace std;

#define PRODUCT_ID_POS 0
#define STORE_ID_POS 1
#define PRICE_POS 2

void printAllProductsAllBuyOptions(vector< vector< vector< int > > > all_products_buy_options);
void printProductAllBuyOption(vector< vector< int > > product_buy_options);
void printProductBuyOption(vector< int > buy_option);

void getBestBuyOptions(vector< vector< vector< int > > > all_products_buy_options, vector< vector< int > > *best_buy_options);
void getProductBestBuyOption(vector< vector< int > > product_buy_options, vector< int > *best_buy_option);
void printBestBuyOption(vector< vector< int > > best_buy_options);

int main()
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

    printAllProductsAllBuyOptions(all_products_buy_options);

    vector< vector< int > > best_buy_options;
    getBestBuyOptions(all_products_buy_options, &best_buy_options);
    cout << endl <<"Best products buy options:" << endl;
    printProductAllBuyOption(best_buy_options);
    return 0;
}

void printAllProductsAllBuyOptions(vector< vector< vector< int > > > all_products_buy_options)
{
    cout << "All products buy options:" << endl;
    for_each(all_products_buy_options.begin(), all_products_buy_options.end(), printProductAllBuyOption);
}

void printProductAllBuyOption(vector< vector< int > > product_buy_options)
{
    for_each(product_buy_options.begin(), product_buy_options.end(), printProductBuyOption);
}

void printProductBuyOption(vector< int > buy_option)
{
    cout << "Buy option:" << endl;
    cout << "\tproduct_id: " << buy_option[0] << endl;
    cout << "\tstore_id: " << buy_option[1] << endl;
    cout << "\tprice: " << buy_option[2] << endl;
}

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
