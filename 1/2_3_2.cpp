#include<iostream>
int main()
{
    int a = 10;
    int *ptr = &a;
    int *p1 = ptr;
    
    std::cout << &a << std::endl;
    std::cout << &ptr << std::endl;
    std::cout << &p1 << std::endl; //其地址是不同的
    std::cout << ptr << std::endl; //指向的地址
    std::cout << p1 << std::endl; //指向的地址
    std::cout << *p1 << std::endl; // 用指针访问对象
    *p1 = 11; // 解引用可以直接赋值 当然 指针不能定义就直接指向常量
    std::cout << a << std::endl;
}