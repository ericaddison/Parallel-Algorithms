#include <future>
#include <iostream>
#include <vector>

int main() {
  std::vector<std::future<int>> futures;

  for(int i = 0; i < 10; ++i) 
    futures.push_back (std::async([](int m) {return 2 * m;} , i));
  
  //retrive and print value stored in  future
  for(auto &e : futures) 
    std::cout << e.get() << std::endl;

  return 0;
}
