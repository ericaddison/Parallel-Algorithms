package main

import "fmt"

func reduce(items []int, c chan int) {
    sum := 0
    if len(items) == 1 {
        sum = items[0]
    } else if len(items) >= 2 {
        middle := len(items) / 2
        c1 := make(chan int)
        c2 := make(chan int)

        go reduce(items[:middle], c1)
        go reduce(items[middle:], c2)

        for i := 0; i < 2; i++ {
            select {
                case sum1 := <- c1:
                    sum += sum1
                case sum2 := <- c2:
                    sum += sum2
            }
        }
    }
    c <- sum
}

func main() {
    A := []int{1,2,3}
    c := make(chan int)
    go reduce (A, c)
    sum := <- c
    fmt.Printf("sum is %d\n", sum)
}
