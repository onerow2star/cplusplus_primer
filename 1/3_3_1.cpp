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

    vector<int> v;
    for (size_t i(0); i != 10; ++i) // 这么写试试
        v.push_back(i);
    // vector 能高效的添加元素 预先的设置容量性能可能更差

    for (auto &i : v)
        i *= i;
    for (auto i : v)
        cout << i << " ";
    cout << endl;

    vector<int>::size_type cnt;
    cnt = v.size();


    return 0;
}