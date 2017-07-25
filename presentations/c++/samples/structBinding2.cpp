/* WON'T COMPILE UNLESS YOU USE
   llvm nightly clang++-X.0 snapshot!!! */
   
#include <set>
#include <string>
#include <iomanip>
#include <iostream>
 
int main() {
    std::set<std::string> myset;
    if (auto [iter, success] = myset.insert("Hello"); success) 
        std::cout << "insert is successful. The value is " << std::quoted(*iter) << '\n';
    else
        std::cout << "The value " << std::quoted(*iter) << " already exists in the set\n";
}
