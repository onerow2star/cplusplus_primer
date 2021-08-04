#include <iostream>
#include <vector>
using namespace std;
int main()
{
    int ii = 3.11112 + 3; // 隐式转换
    double di = 3.11112 + 3;

    bool flag;
    char cv;
    short sv;
    unsigned short usv;
    int iv;
    unsigned uiv = -1;
    long lv = 1;
    unsigned long ulv;
    float fv;
    double dv = 0.0;

    3.14159L + 'a'; // 'a' int -> long double
    dv + iv; // int -> double
    dv + fv; // float -> double
    iv = dv; // double -> int
    flag = dv; // 0 或非0 0.0 也是0
    cv + fv; // char -> int int -> float
    sv + cv; // 都提升为int
    cv + lv; // cv -> long
    iv + ulv; // i -> unsigned long
    usv + iv; // 根据unsigned short 和 int所占大小判定 int
    uiv + lv; // 根据unsigned int 和 long大小判定 unsigned long

    //先提升再转换 超过会溢出




}