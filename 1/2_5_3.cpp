#include <iostream>

int main()
{
    int a = 3, b = 4;
    decltype (a == b) d = 2; 
    decltype (a = b) e = a; // int
    std::cout << d << std::endl;


}