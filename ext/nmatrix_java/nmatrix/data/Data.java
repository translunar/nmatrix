// /////////////////////////////////////////////////////////////////////
// // = NMatrix
// //
// // A linear algebra library for scientific computation in Ruby.
// // NMatrix is part of SciRuby.
// //
// // NMatrix was originally inspired by and derived from NArray, by
// // Masahiro Tanaka: http://narray.rubyforge.org
// //
// // == Copyright Information
// //
// // SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
// // NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
// //
// // Please see LICENSE.txt for additional copyright notices.
// //
// // == Contributing
// //
// // By contributing source code to SciRuby, you agree to be bound by
// // our Contributor Agreement:
// //
// // * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
// //
// // == data.java
// //
// // Source file for dealing with data types.

public class Data{
  public void rubyobj_to_jval(){

  }
  public void rubyval_to_jval(){

  }
  public static float[] castDoubleToFloat(double[] elements){
  	float[] newElements = new float[elements.length];
  	for(int i=0; i < elements.length; i++){
  		newElements[i] = (float)elements[i];
  	}
  	return newElements;
  }

  public static void interpret_cast(Object element){
  	int flag;
  	if(WrapperType.isWrapperType(element.getClass()) == true){
  		String className = element.getClass().getSimpleName();
  		switch(className){
  			case "Byte":
  				flag = 0;
					break;
				case "Short":
					flag = 1;
					break;
				case "Integer":
					flag = 2;
					break;
				// case INT32:
				// 	flag = "INT32";
				// 	break;
				case "Long":
					flag = 3;
					break;
				case "Float":
					flag = 4;
					break;
				case "Double":
					flag = 5;
					break;
				// case COMPLEX64:
				// 	flag = null
				// 	break;
				// case COMPLEX128:
				// 	flag = null
				// 	break;
				// case RUBYOBJ:
				// 	flag = null
				// 	break;

  		}
  	}
  }
}