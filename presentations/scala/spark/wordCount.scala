val textFile = sc.textFile("warandpeace.txt")
val splitFile = textFile.flatMap(line => line.split(" "))
val counts = splitFile.map(word=>(word,1)).reduceByKey(_ + _)
counts.saveAsTextFile("counts.txt")

println(counts.count)
counts.sortBy(_._2, ascending=false).take(10).foreach(s=>println(s))
