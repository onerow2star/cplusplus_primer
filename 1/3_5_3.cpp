#include <iostream>
#include <string>
using namespace std;
int main()
{
    string nums[] = {"one", "tow", "three"};
    string *p = &nums[0]; // p指向nums的第一个元素   
    //
    string *p1 = nums; // 等价*p = &nums[0]

    int ia[] = {0, 2, 3};
    auto ia2(ia); // ia2 是整型指针 指向ia第一个元素
    auto ia3(&ia[0]); //等价

    decltype(ia) ia4 = {1, 2, 3}; // decltype(ia) 是int[3]
    ++p;
    string *e = &nums[3];

    for (string *b = nums; b != &nums[3]; ++b)
    {
        cout << *b << endl;
    }

    int *beg = begin(ia), *last = end(ia);
    while(beg != last && *beg >= 0)
        ++beg;
    constexpr size_t sz = 1;
    int *ip1 = ia + sz; // ia 转换为首指针
    int *ip2 = ia + 2;

    auto n = end(ia) - begin(ia);
    
    int k  = *ia + 4; // k 4
    int k1 = *(ia+1); // k 2

    int *k2 = &ia[1];
    int k3 = k2[1];
    int k4 = k2[-1];

    cout << k3;
    cout << k4;


}