#include <iostream>
int main()
{
    int err = 0;
    int *const cur = &err; 
    err = 1;
    std::cout << *cur << std::endl; 
    // cur  常量指针
    const double pi = 3.14159;
    // double *const pip = &pi; // error
    const double *const pip = &pi; // 指向常量对象的常量指针
    std::cout << *pip << std::endl; 
}