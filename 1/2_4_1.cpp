#include <iostream>
int main()
{
    int a = 10;
    const int i = a;
    // 不改变对象均可对其赋值
    int c = i;

    int c1 = 1;
    int &r = c1;
    const int &r1 = c1; // 允许绑定普通对象
    const int &r2 = 12; // 正确
    const int &r3 = r1 * 2; //正确
    // int &r4 = r1 * 2; // 错误
    // int &r5 = 12; // 错误 
    // 非常量引用必须为左值
    r = 2;
    std::cout << r1 << std::endl;
    std::cout << r2 << std::endl;
    std::cout << r3 << std::endl; //c1 修改了 r1 未修改
}