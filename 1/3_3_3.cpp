#include <iostream>
#include <vector>
#include <string>
using namespace std;
int main()
{
    vector<int> v = {0, 1, 2, 3, 4, 5};
    for (auto &i : v)
        i *= i;
    for (auto i : v)
        cout << i << " ";
    cout << endl;

    vector<int>::size_type cnt;
    cnt = v.size();

    // 只要对象不是常量 就可以下标赋值
    vector<unsigned> scores(11, 0);
    unsigned grade; 
    while (cin >> grade)
        if (grade <= 100) // 使用下标 明确下标在合理的范围内
            ++scores[grade/10];

    // 不能使用下标添加元素 空的vector不能直接调用 添加使用push_back()
    // 下标只能访问已经存在的元素
    // 保证下标合法尽可能使用范围for语句
    return 0;
}