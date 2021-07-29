#include<iostream>
using namespace std;
int main()
{
    int a = 10;
    int *ptr = &a;
    int *&new_ptr = ptr; // 对指针的引用
    cout << &ptr << " " << &new_ptr <<endl;
    return 0;
}