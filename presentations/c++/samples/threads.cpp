#include <iostream>
#include <thread>

static const int num_threads = 10;

// function will be called from a thread
void call_from_thread(int tid) {
    std::cout << "Launched by thread " << tid << std::endl;
}

int main() {
    std::thread t[num_threads];

    //Launch group of threads
    for (int i = 0; i < num_threads; ++i) 
        t[i] = std::thread(call_from_thread, i);

    std::cout << "Launched from the main\n";

    //Join threads with main
    for (int i = 0; i < num_threads; ++i) 
        t[i].join();

    return 0;
}
