def modFunc(x:Int ) = {
  x%2==0
}

val func = modFunc(_)
val list = List range(1, 10)
val filteredList = list.filter(func)


println(filteredList)


// a lambda function
val func2 = (x:Int) => x+x
func2(10)

// map reduce
val l = List(1,2,3,4,5)
l.map( x => x*2 )
l.reduce( (a,b) => a+b )
