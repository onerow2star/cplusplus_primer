#include<iostream>
int main()
{
    int a = 10;
    int *ptr = &a;
    int *&new_ptr = ptr; // 对指针的引用
    std::cout << &ptr << " " << &new_ptr << std::endl;
    return 0;
}