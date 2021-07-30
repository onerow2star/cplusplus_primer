#include <iostream>

constexpr int i = 2; 
int j = 0;
// i j 必须定义于函数体外

int main()
{
    const int *p = nullptr;
    constexpr int *q = nullptr; // 指向整数的常量指针
    
    constexpr const int *p1 = &i; // 常量指针 指向常量
    constexpr int *p2 = &j; // 常量指针

    const int *const pi1 = &i; // 指向常量对象的常量指针
    int *const pi1 = &j; // 指向常量对象的常量指针

}