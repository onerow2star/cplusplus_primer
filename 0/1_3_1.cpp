#include <iostream>
using namespace std;
/*
 * 这
 * 个
 * 函
 * 数
 */
// /**/ 不能嵌套 单行里面可以 忽略嵌套
int main()
{
    // 提示输入
    cout << "Enter two numbers:" << endl;
    int v1 = 0, v2 = 0;     // 保存变量
    cin >> v1 >> v2;        // 输入变量
    cout << "The sum of " << v1 << " and " << v2 << " is " << v1 + v2 << endl;
    return 0;
}