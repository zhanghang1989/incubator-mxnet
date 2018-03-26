/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnetexamples.inferexample.objectdetector

import ml.dmlc.mxnet.{DType, Shape, DataDesc}
import ml.dmlc.mxnet.infer._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import java.nio.file.{Files, Paths}

class SSDClassifierExample {
  @Option(name = "--model-path-prefix", usage = "the input model directory and prefix of the model")
  private val modelPathPrefix: String = "/model/ssd_resnet50_512"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath: String = "/images/dog.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir: String = "/images/"
}

object SSDClassifierExample {

  private val logger = LoggerFactory.getLogger(classOf[SSDClassifierExample])
  private type SSDOut = (String, Array[Float])

  def runObjectDetectionSingle(modelPathPrefix: String, inputImagePath: String):
  IndexedSeq[IndexedSeq[(String, Array[Float])]] = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 512, 512)
    // ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
    val outputShape = Shape(1, 6132, 6)
    val inputDescriptors = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))
    val img = ImageClassifier.loadImageFromFile(inputImagePath)
    val objDetector = new ObjectDetector(modelPathPrefix, inputDescriptors)
    val output = objDetector.imageObjectDetect(img, Some(3))

    output
  }

  def runObjectDetectionBatch(modelPathPrefix: String, inputImageDir: String):
  IndexedSeq[IndexedSeq[(String, Array[Float])]] = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 512, 512)
    // ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
    val outputShape = Shape(1, 6132, 6)
    val inputDescriptors = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))
    val imgList = ImageClassifier.loadInputBatch(inputImageDir)
    val objDetector = new ObjectDetector(modelPathPrefix, inputDescriptors)
    val outputList = objDetector.imageBatchObjectDetect(imgList, Some(1))
    outputList
  }

  def main(args: Array[String]): Unit = {
    val inst = new SSDClassifierExample
    val parser : CmdLineParser = new CmdLineParser(inst)
    parser.parseArgument(args.toList.asJava)
    val baseDir = System.getProperty("user.dir")
    val mdprefixDir = baseDir + inst.modelPathPrefix
    val imgPath = baseDir + inst.inputImagePath
    val imgDir = baseDir + inst.inputImageDir
    if (!checkExist(Array(mdprefixDir + "-symbol.json", imgDir, imgPath))) {
      logger.error("Model or input image path does not exist")
      sys.exit(1)
    }

    try {
      val inputShape = Shape(1, 3, 512, 512)
      val outputShape = Shape(1, 6132, 6)

      val width = inputShape(2)
      val height = inputShape(3)
      var outputStr : String = "\n"

      val output = runObjectDetectionSingle(mdprefixDir, imgPath)


      for (ele <- output) {
        for (i <- ele) {
          outputStr += "Class: " + i._1 + "\n"
          val arr = i._2
          outputStr += "Probabilties: " + arr(0) + "\n"
          val coord = Array[Float](
            arr(1) * width, arr(2) * height,
            arr(3) * width, arr(4) * height
          )
          outputStr += "Coord:" + coord.mkString(",") + "\n"
        }
      }
      logger.info(outputStr)

      val outputList = runObjectDetectionBatch(mdprefixDir, imgDir)

      outputStr = "\n"
      for (idx <- outputList.indices) {
        outputStr += "*** Image " + (idx + 1) + "***" + "\n"
        for (i <- outputList(idx)) {
          outputStr += "Class: " + i._1 + "\n"
          val arr = i._2
          outputStr += "Probabilties: " + arr(0) + "\n"
          val coord = Array[Float](
            arr(1) * width, arr(2) * height,
            arr(3) * width, arr(4) * height
          )
          outputStr += "Coord:" + coord.mkString(",") + "\n"
        }
      }
      logger.info(outputStr)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
    sys.exit(0)
  }


  def checkExist(arr : Array[String]) : Boolean = {
    var exist : Boolean = true
    for (item <- arr) {
      exist = Files.exists(Paths.get(item)) && exist
      if (!exist) {
        logger.error("Cannot find: " + item)
      }
    }
    exist
  }

}
