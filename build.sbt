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


assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}
// Here, `libraryDependencies` is a set of dependencies, and by using `+=`,
// we're adding the cats dependency to the set of dependencies that sbt will go
// and fetch when it starts up.
// Now, in any Scala file, you can import classes, objects, etc, from cats with
// a regular import.

// TIP: To find the "dependency" that you need to add to the
// `libraryDependencies` set, which in the above example looks like this:

// "org.typelevel" %% "cats-core" % "1.0.1"

// You can use Scaladex, an index of all known published Scala libraries. There,
// after you find the library you want, you can just copy/paste the dependency
// information that you need into your build file. For example, on the
// typelevel/cats Scaladex page,
// https://index.scala-lang.org/typelevel/cats, you can copy/paste the sbt
// dependency from the sbt box on the right-hand side of the screen.

// IMPORTANT NOTE: while build files look _kind of_ like regular Scala, it's
// important to note that syntax in *.sbt files doesn't always behave like
// regular Scala. For example, notice in this build file that it's not required
// to put our settings into an enclosing object or class. Always remember that
// sbt is a bit different, semantically, than vanilla Scala.

// ============================================================================

// Most moderately interesting Scala projects don't make use of the very simple
// build file style (called "bare style") used in this build.sbt file. Most
// intermediate Scala projects make use of so-called "multi-project" builds. A
// multi-project build makes it possible to have different folders which sbt can
// be configured differently for. That is, you may wish to have different
// dependencies or different testing frameworks defined for different parts of
// your codebase. Multi-project builds make this possible.

// Here's a quick glimpse of what a multi-project build looks like for this
// build, with only one "subproject" defined, called `root`:

// lazy val root = (project in file(".")).
//   settings(
//     inThisBuild(List(
//       organization := "ch.epfl.scala",
//       scalaVersion := "2.12.4"
//     )),
//     name := "hello-world"
//   )

// To learn more about multi-project builds, head over to the official sbt
// documentation at http://www.scala-sbt.org/documentation.html

