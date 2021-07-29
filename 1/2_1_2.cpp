#include <iostream>
int main()
{
    unsigned char c = -1; // 负数会转化为无符号数

    // unsigned char c = 256; // 不在取值范围内，编译器会自己报错
    // int i = c;
    // std::cout << i << std::endl;
    unsigned u = -1; // 2^32 - 1
    int i = 1;
    std::cout << u * i << std::endl; // 计算既有有符号数又有无符号数时，先将有符号数转化为无符号数再进行计算
    unsigned u1 = 10, u2 = 20;
    std::cout << u2 - u1 << std::endl;
    std::cout << u1 - u2 << std::endl; // 计算结果为负 同样转化为无符号数
    // 同理 对 --u 操作时 u永远不会小于0
}