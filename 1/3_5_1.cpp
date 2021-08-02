#include <iostream>
#include <string>
using namespace std;
int main()
{
    unsigned cnt = 42; // 不是常量表达式
    constexpr unsigned sz = 3;
    int arr[10];
    int *parr[sz];
    // string bad[cnt]; // error
    // string strs[get_size()]; // get_size是常量表达式成立
    const unsigned size = 2;
    int ia[size] = {0, 1};
    int a1[] = {0, 1, 2};
    int a3[5] = {0, 1, 2}; // {0, 1, 2. 0, 0};
    string a4[3] = {"hi", "bye"}; // {"hi", "bye", ""}
    // int a5[2] = {0, 1, 2}; // error
    char ca1[] = {'c', '+', '+'}; // ca1维度为3
    char ca2[] = {'c', '+', '+', '\0'};
    char ca3[] = "c++"; // ca3维度为4
    // char ca4[3] = "c++"; // error 没有空间

    // int na = a1; // error
    int a2[3];
    // a2 = a1; // error
    
    int *p[10];
    int (*pa)[10];
    int (&ra)[3] = a1; // 引用必须初始化 且必须指定维度
    int *(&rap)[10] = p;
}