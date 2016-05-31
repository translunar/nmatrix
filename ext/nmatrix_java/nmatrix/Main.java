// package com.sciruby.nmatrix;
// Only for PUTS DEBUGGING
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;

public class Main{

	public static void main(String[] args){
		// System.out.println("Hello jnmatrix");
		JNMatrix a = new JNMatrix(2, new double[]{2,3,4,5}, "FLOAT32", "DENSE_STORE");
		JNMatrix b = new JNMatrix(2, new double[]{2.2,3,4,5}, "FLOAT32", "DENSE_STORE");
		JNMatrix c = a.add(b);
		// float[] abc = new float[4];
		double[] abc = c.realArray.toArray();

		System.out.println(abc);
		for(int i=0; i < abc.length; i++){
  		System.out.println(abc[i]);
  	}
	}

	// public static float[] castDoubleToFloat(double[] elements){
 //  	float[] newElements = new float[elements.length];
 //  	for(int i=0; i < elements.length; i++){
 //  		newElements[i] = (float)elements[i];
 //  	}
 //  	return newElements;
 //  }
}	