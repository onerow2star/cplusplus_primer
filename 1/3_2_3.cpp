#include <iostream>
#include <string>
// #include <cctype> // string包含过了
using std::string;
using std::cin;
using std::cout;
using std::endl;
int main()
{
    string s("Some string!!!");
    decltype (s.size()) punct_cnt = 0;
    for (auto c : s)
        if (ispunct(c))
            ++punct_cnt;
    cout << punct_cnt << " punctuation characters in " << s <<endl;
    
    for (auto &c : s) // c是引用 不是临时变量
        c = tolower(c);
    
    if (!s.empty())
        s[0] = toupper(s[0]);
    cout << s << endl;

    // decltype(s.size()) 保证下标合法性 >= 0
    for (decltype(s.size()) i = 0; i != s.size() && !isspace(s[i]); ++i)
        s[i] = toupper(s[i]);
    cout << s << endl; 

    const string hexdigits = "0123456789ABCDEF";
    cout << "Enter a series of numbers between 0 and 15"
         << " spearated by spaces. Hit ENTER when finished: "
         << endl;
    string result;
    string::size_type n;
    while(cin >> n)
        if (n < hexdigits.size())
            result += hexdigits[n];
    cout << "Your hex number is: " << result << endl;
    return 0;
}