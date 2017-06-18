// this is a class
// the class parameters (x,y) have default arguments
class Point(
    val x: Double = 0.0, val y: Double = 0.0
) {
  import Point._

  def this() = this(0.0, 0.0)   // defining a no-arg ctor

  def distanceToPoint(other: Point) =
    distanceBetweenPoints(x, y, other.x, other.y)
}

// this is a companion object to the Point class
// it is a singleton object
object Point {
  def distanceBetweenPoints(x1: Double, y1: Double,
      x2: Double, y2: Double) = {
    math.hypot(x1 - x2, y1 - y2)
  }
}

object PointDemo extends App {
  val point1 = new Point(1,2)
  val point2 = new Point(y=3) // named parameters
  println("point1 = (" + point1.x + ", " + point1.y + ")")
  println("point2 = (" + point2.x + ", " + point2.y + ")")
  println("Distance between points: " + point1.distanceToPoint(point2))
}
