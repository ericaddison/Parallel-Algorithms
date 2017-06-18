// simpleThread.scala
// a VERY simple example of threading in scala

println("Creating a thread...")

val thread = new Thread{
  override def run{
    println("Thread starting!")
    Thread.sleep(2000)
    println("Thread done!")
  }
}

thread.start()
println("Thread created and started")
