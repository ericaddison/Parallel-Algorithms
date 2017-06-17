def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/1000000.0 + "ms\n\n")
    result
}

val n = 1000000

// a sequential array
val array = (1 to n).toArray

// a parallel array
val parArray = (1 to n).toArray.par

println("\n\nSequential reduce")
println("------------------")
time[Int](array.reduce( (a,b) => a+b))

println("Parallel reduce")
println("------------------")
time[Int](parArray.reduce( (a,b) => a+b))
