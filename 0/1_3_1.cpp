#include <iostream>
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
    std::cout << "Enter two numbers:" << std::endl;
    int v1 = 0, v2 = 0;     // 保存变量
    std::cin >> v1 >> v2;        // 输入变量
    std::cout << "The sum of " << v1 << " and " << v2 << " is " << v1 + v2 << std::endl;
    return 0;
}