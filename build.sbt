// The simplest possible sbt build file is just one line:

scalaVersion := "2.11.8"
// That is, to create a valid sbt build, all you've got to do is define the
// version of Scala you'd like your project to use.

// ============================================================================

// Lines like the above defining `scalaVersion` are called "settings" Settings
// are key/value pairs. In the case of `scalaVersion`, the key is "scalaVersion"
// and the value is "2.12.4"

// It's possible to define many kinds of settings, such as:

name := "gradientgmm"
organization := "com.github.nestorsag.gradientgmm"
version := "1.0"

// Note, it's not required for you to define these three settings. These are
// mostly only necessary if you intend to publish your library's binaries on a
// place like Sonatype or Bintray.


// Want to use a published library in your project?
// You can define other libraries as dependencies in your build like this:
libraryDependencies += "org.typelevel" %% "cats-core" % "1.0.1"

//test dependencies
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.5" % "test"

libraryDependencies  ++= Seq(

  "com.github.fommil.netlib" % "all" % "1.1.2",

  "org.typelevel" %% "cats-core" % "1.0.1",
  // other dependencies here
  "org.scalanlp" % "breeze_2.11" % "0.13.2",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes. 
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" % "breeze-natives_2.11" % "0.13.2",
  // the visualization library is distributed separately as well. 
  // It depends on LGPL code.

  "org.scalanlp" % "breeze-viz_2.11" % "0.13.2",

  // spark dependencies
  "org.apache.spark" % "spark-mllib_2.11" % "2.3.1",

  "org.apache.spark" % "spark-core_2.11" % "2.3.1"

  //logging dependencies
  //"ch.qos.logback" % "logback-classic" % "1.2.3",

  //"com.typesafe.scala-logging" %% "scala-logging" % "3.9.0"


)

dependencyOverrides ++= Seq(
  
  "io.netty" % "netty" % "3.9.9.Final",

  "commons-net" % "commons-net" % "2.2",

  "commons-io" % "commons-io" % "2.4",

  "com.google.guava" % "guava" % "11.0.2",

  "com.google.code.findbugs" % "jsr305" % "3.0.2"
)


resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)


// assemblyMergeStrategy in assembly := {
//  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//  case x => MergeStrategy.first
// }