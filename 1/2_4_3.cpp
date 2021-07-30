#include <iostream>
int main()
{
    int i = 2;
    int *const p1 = &i; //常量指针 p1 不能修改 顶层const
    const int ci = 42; // 常量整型 ci 不能修改 顶层const
    const int *p2 = &ci; // 指向常量的指针 允许改变p2 底层const
    p2 = &i; // 只不过不允许p2修改i的值 但是可以指向任意数值
    std::cout << *p2 << std::endl;

    const int *const p3 = p2; // 右边顶层 左边底层
    const int &r = ci; // 声明引用的都是底层 
    // 对常量的引用
    // 引用可绑定常量 等价于绑定了一个临时的对象 不可修改
    // 非常量可以转化为常量 反之不行
    // int *p = p3; // error
    p2 = p3; // 可
    // int &r1 = ci; // error
    const int &r2 = i; // 可

}

