#include <iostream>
using namespace std;
// void print(const int*);
void print1(const int a[3])
{
    cout << *a << endl;
}
void print2(const int *a)
{
    cout << *a << endl;
}
// void print(const int[10]);

void print3(int (&a)[3])
{
    cout << *a << endl;
}

int main()
{
    int i = 0, j[2] = {1, 2};
    print1(&i); 
    print2(j);
    // print3(j); error 必须指定相同的维度
}