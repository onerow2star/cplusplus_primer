#include <iostream>
#include <vector>
#include <string>
using namespace std;
int main()
{
    vector<int> v;
    // size_t i = -1; // 32 unsigned long 64 unsigned long long
    for (size_t i(0); i != 10; ++i) // 这么写试试
        v.push_back(i);
    // vector 能高效的添加元素 预先的设置容量性能可能更差
    string word;
    vector<string> text;
    while(cin >> word)
        text.push_back(word);

    return 0;
}