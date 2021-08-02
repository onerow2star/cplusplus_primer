#include <iostream>
using namespace std;
int main()
{
    const char ca1[] = "A s";
    const char ca2[] = "A d";
    // if (ca1 < ca2) // 比较的ca1 ca2的指向首字母元素的地址
    if(strcmp(ca1, ca2) < 0)
        cout << " " << endl;
    return 0;
}