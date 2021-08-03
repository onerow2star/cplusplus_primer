#include <iostream>
using namespace std;
int main()
{
    unsigned char bits = 0233; // char占8位 10011011
    bits << 8; // 转为int型 32位 想左移动8位
    //00000000 00000000 10011011 00000000
    bits << 31;  //向右移动32位
    //10000000 00000000 00000000 00000000 
    bits >> 3;
    //00000000 00000000 00000000 00010011
    ~bits; // 先提升int 
    //11111111 11111111 11111111 01101000
    unsigned char b1 = 0145;
    unsigned char b2 = 0257;
    b1 & b2; 
    b1 | b2;
    b1 ^ b2; // 相同为0 不同为1

    // 学生考试 一个位一个学生 30个
    unsigned long quiz1 = 0; // 至少32位
    quiz1 |= 1UL << 27;// 生成一个值 第27位为1
    quiz1 &= ~(1UL << 27);// 第27位为0
    bool status = quiz1 & (1UL << 27);// 判断其是否通过测验
    
    // 移位运算符满足左结合律
    cout << "hi" << "there" <<endl;
    ( ( cout << "hi" ) << "there" ) << endl;

    cout << 42 + 10; // + 优先级高
    cout << (42 < 10);
    // cout << 42 < 10; 比较(cout << 42)返回cout 和 42
}