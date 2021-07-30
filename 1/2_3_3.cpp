#include<iostream>
int main()
{
    int a = 10;
    int *ptr = &a;
    int *p1 = ptr;
    int **p2 = &p1;
    
    std::cout << p1 << std::endl;
    std::cout << &p1 << std::endl;
    std::cout << p2 << std::endl;
    std::cout << *p2 << std::endl;
    std::cout << **p2 << std::endl;
}