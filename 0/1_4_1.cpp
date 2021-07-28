#include <iostream>
int main()
{
    int sum = 0, val = 1;
    // val < 10 
    while(val <= 10)
    {
        sum += val;
        ++val; // val加1 ++i 更有效率
    }
    std::cout << "Sum of 1 to 10 inclusive is " << sum << std::endl;
    return 0;
}