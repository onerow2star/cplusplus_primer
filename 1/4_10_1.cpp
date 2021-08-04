#include <iostream>
#include <vector>
using namespace std;
int main()
{
    vector<int> ivec = {0, 1, 2};
    vector<int>::size_type cnt = ivec.size();
    for (vector<int>::size_type ix = 0; ix != ivec.size(); ++ix, --cnt)
    {
        ivec[ix] = cnt;
        cout << ivec[ix] << endl;;
    }
}