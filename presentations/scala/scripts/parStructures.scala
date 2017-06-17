def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/1000000.0 + "ms\n\n")
    result
}

val n = 6000000

// a sequential array
val array = (1 to n).toArray

// a parallel array
val parArray = (1 to n).toArray.par

println("\n\nParallel reduce")
println("------------------")
time[Int](parArray.reduce( (a,b) => a+b))

println("Sequential reduce")
println("------------------")
time[Int](array.reduce( (a,b) => a+b))

println("Parallel reduce (again)")
println("------------------")
time[Int](parArray.reduce( (a,b) => a+b))

println("Sequential reduce (again)")
println("------------------")
time[Int](array.reduce( (a,b) => a+b))
