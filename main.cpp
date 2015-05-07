#include <iostream>
#include <vector>
#include <algorithm> // for_each function

using namespace std;


void printAllProductsAllBuyOptions(vector< vector< vector< int > > > all_products_buy_options);
void printProductAllBuyOption(vector< vector< int > > product_buy_options);
void printProductBuyOption(vector< int > buy_option);

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

    product_1_buy_options.push_back(product_1_store_1_buy_option);
    product_1_buy_options.push_back(product_1_store_2_buy_option);
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
