#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main()
{
    vector<int> v1{1, 1, 2, 3, 4, 5, 6, 7};
    auto mid = v1.begin() + v1.size() / 2;
    auto it = v1.begin() + 3;
    if(it < mid)
        cout << "?" <<endl;
    auto sought = 9;
    auto beg = v1.begin(), end = v1.end();
    while(mid != end && *mid != sought)
    {
        if(sought < *mid)
            end = mid;
        else
            beg = mid + 1;
        mid = beg + (end - beg) / 2;
    }
    if(*mid == sought)
        cout << *mid << endl;
    else
        cout << "Not found" << endl;
}