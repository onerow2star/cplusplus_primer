#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main()
{
    string s("some string");
    if (s.begin() != s.end())
    {
        auto it = s.begin();
        *it = toupper(*it);
    }
    cout << s << endl;
    for (auto it = s.begin(); it != s.end() && !isspace(*it); ++it)
    {
        *it = toupper(*it);
    }
    cout << s << endl;
    
    vector<string> text;
    text.push_back(s);
    // 泛型编程 C++ 习惯使用 != 因为所有容器都定义了 == != 大多未定义 < 运算符
    for (auto it = text.cbegin(); it != text.cend() && !it->empty(); ++it)
    {
        cout << *it << endl;
    }
    
}