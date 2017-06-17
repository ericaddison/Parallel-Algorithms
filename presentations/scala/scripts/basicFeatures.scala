
// basic assignment
var x = 10  // a variable
val y = 20  // a value

x = 20      // ok
// y = 100     // ERROR! immutable!

// function definition
def square(x:Double): Double = {
  x*x
}
square(2)   // = 4

// control flow
for(a <- 1 to 10)
  println(a)

for(a <- (1 to 6).map(a=>math.Pi/a))
  println("%1.3f".format(math.cos(a)))

for(a <- 1 to 3; b <- 3 to 6)
  yield(a,b)

if (1<2) {
  println("yup")
}

var j = 0
while(j<10){
  println(j)
  j = j+1
}

// no switch! use match
val i = 3
i match {
  case 1  => println("Red")
  case 2  => println("Green")
  case 3  => println("Blue")
  case default  => println("error: " + default.toString)
} // prints "Blue"
