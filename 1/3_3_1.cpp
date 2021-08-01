#include <iostream>
#include <vector>
#include <string>
using namespace std;
int main()
{
    // vector<int> v1 = 10; // error 必须直接初始化形式
    vector<int> v1(10); // 10个为0的int
    vector<int> v2{10}; // 1个元素 10 
    vector<int> v3(10, 1); 
    vector<int> v4{10, 1}; 

    // 花括号尽量表现成列表初始化的形式
    // vector<string> v5("hi"); // error
    vector<string> v6{"hi"}; // 1个元素 10 
    vector<string> v7(10, "hi"); 
    vector<string> v8{10};  // 10个空串
    vector<string> v9(10); 
    return 0;
}