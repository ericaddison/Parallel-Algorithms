import util.Random.nextInt
println()

val n = 10
val x = 5
val y = 10

val A = Seq.fill(n)(nextInt%10+10).par
println("A = " + A)

val p = (a:Int) => {if ((a>=x)&&(a<=y)) 1 else 0}
val B = A.map(p)
println("B = " + B)

val C = B.scanLeft(0)(_+_)
println("C = " + C)

val size = C(n)
val E = Array.fill[Int](size)(0).par

B.zipWithIndex.foreach(s=> if(s._1==1) E(C(s._2)) = A(s._2))
println("E = " + E)

// or just use filter :)
val p2 = (a:Int) => {(a>=x)&&(a<=y)}
val F = A.filter(p2)
println("F = " + F)

println()
