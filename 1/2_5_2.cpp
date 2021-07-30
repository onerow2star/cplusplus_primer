#include <iostream>

int main()
{
    const int ci = 10;
    // auto &r1 = 10; // error
    auto &r = ci, *p = &ci;
    // 顶层const int *const &r  *p
    int i = 1;
    // auto &r = i, *p2 = &ci; // error

}