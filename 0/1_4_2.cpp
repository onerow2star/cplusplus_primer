#include<iostream>
int main()
{
    int sum = 0;
    for (int val = -100; val <= 100; ++val)
        sum += val;
    std::cout << "Sum of 1 to 10 inclusive is " << sum << std::endl;
    return 0;
}