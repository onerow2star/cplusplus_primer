#include <iostream>
#include <string>
#include <windows.h>
using namespace std;

const string &mainp()
{
    string ret;
    ret = "23";
    if (!ret.empty())
        return ret; // error 
    else 
        return "Empty"; 
        //该对象也是局部临时的对象 
    // 这两条return语句都指向了不再可用的局部空间

}

int main()
{
    const string ps = mainp(); // error: taking address of temporary [-fpermissive
    cout << ps << endl; 
    // Sleep(1);
    // cout << ps << endl;
    return 0;
}