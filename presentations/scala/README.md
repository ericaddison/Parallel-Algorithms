#Scala Examples
This directory contains example Scala code as shown in the Scala presentation. There are four subdirectories:
- src: compilable Scala code
- class: compiled class files
- scripts: Scala script files to be run by the interpreter
- spark: an example Scala code using Spark 

##Compiled Scala
Scala code is compiled using the `scalac` compiler. A `Makefile` is provided for easy compiling. After the Scala code is compiled, you can run a class that extends `App` by moving into the same directory and running `scala HelloWorld`, for example.

##Interpreted Scala
The script files can be run by the Scala interpreter either from the command line with
```
scala <file>
```

or from within the interactive Scala REPL with
```
:load <file>
```

##Spark
Spark must be installed to run the Spark example. There are many websites demonstrating how to install Spark, for example: [here](https://www.tutorialspoint.com/apache_spark/apache_spark_installation.htm).

The example can be run from within the `spark-shell` REPL using the same `:load` command as above. (Make sure you are either in the same directory as the text file or you insert the full path!)
