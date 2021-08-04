#include <iostream>
#include <vector>
#include <string>
using namespace std;
int main()
{
    int i = 10, j = 3;
    double d0 = i / j;// 3
    double d1 = static_cast<double>(i) / j; // 3.33333
    
    void *p = &d0;
    double *dp = static_cast<double*>(p);

    const char *pc = "1";
    // char *p = static_cast<char*>(pc); // error 
    char *p1 = const_cast<char*>(pc); // 通过p写值是未定义的行为
    static_cast<string>(pc); // 字符串字面值 
    // const_cast<string>(pc);  error const_cast只改变常量属性

    char a ='a';
    // reinterpret_cast 仅仅是复制 n 的比特位到 d，因此d 包含无用值。
    int ic=reinterpret_cast<int&>(a);
    cout << ic;
   

}
