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

package org.apache.mxnet

import org.apache.mxnet.init.Base._
import org.apache.mxnet.utils.OperatorBuildUtils

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

private[mxnet] class AddNDArrayFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.addDefs
}

private[mxnet] class AddNDArrayAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.typeSafeAPIDefs
}

private[mxnet] object NDArrayMacro {
  case class NDArrayArg(argName: String, argType: String, isOptional : Boolean)
  case class NDArrayFunction(name: String, listOfArgs: List[NDArrayArg])

  // scalastyle:off havetype
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    impl(c)(annottees: _*)
  }
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typeSafeAPIImpl(c)(annottees: _*)
  }
  // scalastyle:off havetype

  private val ndarrayFunctions: List[NDArrayFunction] = initNDArrayModule()

  private def impl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    val newNDArrayFunctions = {
      if (isContrib) ndarrayFunctions.filter(_.name.startsWith("_contrib_"))
      else ndarrayFunctions.filter(!_.name.startsWith("_contrib_"))
    }

     val functionDefs = newNDArrayFunctions flatMap { NDArrayfunction =>
        val funcName = NDArrayfunction.name
        val termName = TermName(funcName)
        if (!NDArrayfunction.name.startsWith("_") || NDArrayfunction.name.startsWith("_contrib_")) {
          Seq(
            // scalastyle:off
            // (yizhi) We are investigating a way to make these functions type-safe
            // and waiting to see the new approach is stable enough.
            // Thus these functions may be deprecated in the future.
            // e.g def transpose(kwargs: Map[String, Any] = null)(args: Any*)
            q"def $termName(kwargs: Map[String, Any] = null)(args: Any*) = {genericNDArrayFunctionInvoke($funcName, args, kwargs)}".asInstanceOf[DefDef],
            // e.g def transpose(args: Any*)
            q"def $termName(args: Any*) = {genericNDArrayFunctionInvoke($funcName, args, null)}".asInstanceOf[DefDef]
            // scalastyle:on
          )
        } else {
          // Default private
          Seq(
            // scalastyle:off
            q"private def $termName(kwargs: Map[String, Any] = null)(args: Any*) = {genericNDArrayFunctionInvoke($funcName, args, kwargs)}".asInstanceOf[DefDef],
            q"private def $termName(args: Any*) = {genericNDArrayFunctionInvoke($funcName, args, null)}".asInstanceOf[DefDef]
            // scalastyle:on
          )
        }
      }

    structGeneration(c)(functionDefs, annottees : _*)
  }

  private def typeSafeAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*) : c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }

    val newNDArrayFunctions = {
      if (isContrib) ndarrayFunctions.filter(
        func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
      else ndarrayFunctions.filterNot(_.name.startsWith("_"))
    }

    val functionDefs = newNDArrayFunctions map { ndarrayfunction =>

      // Construct argument field
      var argDef = ListBuffer[String]()
      // Construct Implementation field
      var impl = ListBuffer[String]()
      impl += "val map = scala.collection.mutable.Map[String, Any]()"
      ndarrayfunction.listOfArgs.foreach({ ndarrayarg =>
        // var is a special word used to define variable in Scala,
        // need to changed to something else in order to make it work
        val currArgName = ndarrayarg.argName match {
          case "var" => "vari"
          case "type" => "typeOf"
          case default => ndarrayarg.argName
        }
        if (ndarrayarg.isOptional) {
          argDef += s"${currArgName} : Option[${ndarrayarg.argType}] = None"
        }
        else {
          argDef += s"${currArgName} : ${ndarrayarg.argType}"
        }
        var base = "map(\"" + ndarrayarg.argName + "\") = " + currArgName
        if (ndarrayarg.isOptional) {
          base = "if (!" + currArgName + ".isEmpty)" + base + ".get"
        }
        impl += base
      })
      // scalastyle:off
      impl += "org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(\"" + ndarrayfunction.name + "\", null, map.toMap)"
      // scalastyle:on
      // Combine and build the function string
      val returnType = "org.apache.mxnet.NDArray"
      var finalStr = s"def ${ndarrayfunction.name}New"
      finalStr += s" (${argDef.mkString(",")}) : $returnType"
      finalStr += s" = {${impl.mkString("\n")}}"
      c.parse(finalStr).asInstanceOf[DefDef]
    }

    structGeneration(c)(functionDefs, annottees : _*)
  }

  private def structGeneration(c: blackbox.Context)
                              (funcDef : List[c.universe.DefDef], annottees: c.Expr[Any]*)
  : c.Expr[Any] = {
    import c.universe._
    val inputs = annottees.map(_.tree).toList
    // pattern match on the inputs
    val modDefs = inputs map {
      case ClassDef(mods, name, something, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ClassDef(mods, name, something, q)
      case ModuleDef(mods, name, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ModuleDef(mods, name, q)
      case ex =>
        throw new IllegalArgumentException(s"Invalid macro input: $ex")
    }
    // wrap the result up in an Expr, and return it
    val result = c.Expr(Block(modDefs, Literal(Constant())))
    result
  }


  // Convert C++ Types to Scala Types
  private def typeConversion(in : String, argType : String = "") : String = {
    in match {
      case "Shape(tuple)" | "ShapeorNone" => "org.apache.mxnet.Shape"
      case "Symbol" | "NDArray" | "NDArray-or-Symbol" => "org.apache.mxnet.NDArray"
      case "Symbol[]" | "NDArray[]" | "NDArray-or-Symbol[]" | "SymbolorSymbol[]"
      => "Array[org.apache.mxnet.NDArray]"
      case "float" | "real_t" | "floatorNone" => "org.apache.mxnet.Base.MXFloat"
      case "int" | "intorNone" | "int(non-negative)" => "Int"
      case "long" | "long(non-negative)" => "Long"
      case "double" | "doubleorNone" => "Double"
      case "string" => "String"
      case "boolean" | "booleanorNone" => "Boolean"
      case "tupleof<float>" | "tupleof<double>" | "ptr" | "" => "Any"
      case default => throw new IllegalArgumentException(
        s"Invalid type for args: $default, $argType")
    }
  }


  /**
    * By default, the argType come from the C++ API is a description more than a single word
    * For Example:
    *   <C++ Type>, <Required/Optional>, <Default=>
    * The three field shown above do not usually come at the same time
    * This function used the above format to determine if the argument is
    * optional, what is it Scala type and possibly pass in a default value
    * @param argType Raw arguement Type description
    * @return (Scala_Type, isOptional)
    */
  private def argumentCleaner(argType : String) : (String, Boolean) = {
    val spaceRemoved = argType.replaceAll("\\s+", "")
    var commaRemoved : Array[String] = new Array[String](0)
    // Deal with the case e.g: stype : {'csr', 'default', 'row_sparse'}
    if (spaceRemoved.charAt(0)== '{') {
      val endIdx = spaceRemoved.indexOf('}')
      commaRemoved = spaceRemoved.substring(endIdx + 1).split(",")
      commaRemoved(0) = "string"
    } else {
      commaRemoved = spaceRemoved.split(",")
    }
    // Optional Field
    if (commaRemoved.length >= 3) {
      // arg: Type, optional, default = Null
      require(commaRemoved(1).equals("optional"))
      require(commaRemoved(2).startsWith("default="))
      (typeConversion(commaRemoved(0), argType), true)
    } else if (commaRemoved.length == 2 || commaRemoved.length == 1) {
      val tempType = typeConversion(commaRemoved(0), argType)
      val tempOptional = tempType.equals("org.apache.mxnet.NDArray")
      (tempType, tempOptional)
    } else {
      throw new IllegalArgumentException(
        s"Unrecognized arg field: $argType, ${commaRemoved.length}")
    }

  }


  // List and add all the atomic symbol functions to current module.
  private def initNDArrayModule(): List[NDArrayFunction] = {
    val opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeNDArrayFunction(opHandle.value, opName)
    }).toList
  }

  // Create an atomic symbol function by handle and function name.
  private def makeNDArrayFunction(handle: NDArrayHandle, aliasName: String)
  : NDArrayFunction = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    _LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs)
    val paramStr = OperatorBuildUtils.ctypes2docstring(argNames, argTypes, argDescs)
    val extraDoc: String = if (keyVarNumArgs.value != null && keyVarNumArgs.value.length > 0) {
      s"This function support variable length of positional input (${keyVarNumArgs.value})."
    } else {
      ""
    }
    val realName = if (aliasName == name.value) "" else s"(a.k.a., ${name.value})"
    val docStr = s"$aliasName $realName\n${desc.value}\n\n$paramStr\n$extraDoc\n"
    // scalastyle:off println
    if (System.getenv("MXNET4J_PRINT_OP_DEF") != null
      && System.getenv("MXNET4J_PRINT_OP_DEF").toLowerCase == "true") {
      println("NDArray function definition:\n" + docStr)
    }
    // scalastyle:on println
    val argList = argNames zip argTypes map { case (argName, argType) =>
      val typeAndOption = argumentCleaner(argType)
      new NDArrayArg(argName, typeAndOption._1, typeAndOption._2)
    }
    new NDArrayFunction(aliasName, argList.toList)
  }
}
