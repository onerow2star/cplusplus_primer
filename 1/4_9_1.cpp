#include <iostream>
int main()
{
    int ia[3] = {1, 2};
    constexpr size_t sz = sizeof(ia) / sizeof(*ia);
    std::cout << sz << std::endl;
}