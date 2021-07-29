#include <iostream>
// 仅供说明 内部不宜有与全局变量相同的新变量
int reused= 42;
int main()
{
    std::cout << reused << std::endl;
    int reused = 1;
    std::cout << reused << std::endl;
    // 显示访问全局变量
    std::cout << ::reused << std::endl;
    // 左侧作用域为空是 表示全局作用域
}