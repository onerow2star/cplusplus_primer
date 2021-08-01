#include <iostream>
#include <string>
using std::string;
using std::cin, std::cout, std::endl;
int main()
{
    string s;
    cin >> s; // 输入" Hello World! " 得到 "Hello"
    cout << s << endl;
    string line;
    while(getline(cin, line)) // getline的换行符被丢弃了
        if (!line.empty()) // 遇见非空行
            if (line.size() > 80) // size()返回的是一个 string::size
                cout << line << endl; // line不包含换行符 我们用endl手动加上






    return 0;

}