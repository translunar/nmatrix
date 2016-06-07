import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.ArrayFieldVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.FieldVector;
import org.apache.commons.math3.analysis.function.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.special.Erf;

public class JNMatrix{

  private int[] shape;
  public ArrayRealVector realArray;
  private String dtype_string;
  private String stype_string;
  public JNMatrixTwoD twoDMat;

  private Dense_Storage storage;
  
  public String get_stype_string(){
    return this.stype_string;
  }


  private Stype stype;
  private Dtype dtype;

  // JNMatrix(shape, elements, dtype, stype)
  public JNMatrix(int[] shape, double[] elements, String dtype_string, String stype_string){
    this.shape = shape;
    this.realArray = new ArrayRealVector(elements);
    this.dtype = dtype;
    this.stype_string = dtype_string;
    this.dtype_string = dtype_string;
    if (shape.length == 2){
      this.twoDMat = new JNMatrixTwoD(shape, elements);
    }
    // if(this.checkVectorDimensions(2)){
    //  //we have a double matrix
    // }

    // switch(stype){
    //  case DENSE_STORE:
    //    this.stype_string = "dense";
    //    // (int dim, int[] shape, String dtype, int offset, int count, double[] elements)
    //    // this.storage = new Dense_Storage( 2,new int[]{2,3}, "int", 1, 4, elements);
    //    break;
    //  case LIST_STORE:
    //    this.stype_string = "list";
    //    break;
    //  case YALE_STORE:
    //    this.stype_string = "yale";
    //    break;
    //  default:
    //    this.stype_string ="stype could not be determined";
    //    break;
    // }

    // switch(dtype){
    //  case BYTE:
    //    this.dtype_string = "BYTE";
    //    break;
    //  case INT8:
    //    this.dtype_string = "INT8";
    //    break;
    //  case INT16:
    //    this.dtype_string = "INT16";
    //    break;
    //  case INT32:
    //    this.dtype_string = "INT32";
    //    break;
    //  case INT64:
    //    this.dtype_string = "INT64";
    //    break;
    //  case FLOAT32:
    //    this.dtype_string = "FLOAT32";
    //    break;
    //  case FLOAT64:
    //    this.dtype_string = "FLOAT64";
    //    break;
    //  case COMPLEX64:
    //    this.dtype_string = "COMPLEX64";
    //    break;
    //  case COMPLEX128:
    //    this.dtype_string = "COMPLEX128";
    //    break;
    //  case RUBYOBJ:
    //    this.dtype_string = "RUBYOBJ";
    //    break;
    // }
  }

  // public JNMatrix(int shape, double[] elements, Dtype dtype){
  //  this(shape, elements, dtype, "DENSE_STORE");
  // }

  // public JNMatrix(int shape, double[] elements){
  //  this(shape, elements,  "FLOAT32", "DENSE_STORE");
  // }

  


  public static JNMatrix aret(JNMatrix a){
      return a;
  }

  // ArrayRealVector add(RealVector v)
  // Compute the sum of this vector and v.

  public double[] add(JNMatrix n){
    ArrayRealVector resRealArray =  this.realArray.add(n.realArray);
    return resRealArray.toArray();
  }

  // ArrayRealVector  subtract(RealVector v)
  // Subtract v from this vector.

  public double[] subtract(JNMatrix n){
    ArrayRealVector resRealArray =  this.realArray.subtract(n.realArray);
    return resRealArray.toArray();
  }

  // void addToEntry(int index, double increment)
  // Change an entry at the specified index.

  public void addToEntry(int index, double increment){
    this.realArray.addToEntry(index, increment);
  }


  // ArrayRealVector  append(ArrayRealVector v)
  // Construct a vector by appending a vector to this vector.

  public JNMatrix append(JNMatrix n){
    RealVector resRealArray =  this.realArray.append(n.realArray);
    JNMatrix res = new JNMatrix(this.shape, resRealArray.toArray(), "FLOAT32", "DENSE_STORE");
    return res;
  }


  // RealVector append(double in)
  // Construct a new vector by appending a double to this vector.

  public JNMatrix append(double in){
    RealVector resRealArray =  this.realArray.append(in);
    JNMatrix res = new JNMatrix(this.shape, resRealArray.toArray(), "FLOAT32", "DENSE_STORE");
    return res;
  }


  // RealVector append(RealVector v)
  // Construct a new vector by appending a vector to this vector.
     
    //ignored

  // protected void checkVectorDimensions(int n)
  // Check if instance dimension is equal to some expected value.

    // protected void checkJNMatrixDimensions(int n){
    //  this.realArray.checkVectorDimensions(n);
    // }


  // protected void checkVectorDimensions(RealVector v)
  // Check if instance and specified vectors have the same dimension.

    //ignored

  // ArrayRealVector  combine(double a, double b, RealVector y)
  // Returns a new vector representing a * this + b * y, the linear combination of this and y.

  // ArrayRealVector  combineToSelf(double a, double b, RealVector y)
  // Updates this with the linear combination of this and y.
  // ArrayRealVector  copy()
  // Returns a (deep) copy of this vector.
  // double dotProduct(RealVector v)
  // Compute the dot product of this vector with v.


  // ArrayRealVector  ebeDivide(RealVector v)
  // Element-by-element division.
  public double[] ebeDivide(JNMatrix n){
    ArrayRealVector resRealArray =  this.realArray.ebeDivide(n.realArray);
    return resRealArray.toArray();
  }


  // ArrayRealVector  ebeMultiply(RealVector v)
  // Element-by-element multiplication.
  public double[] ebeMultiply(JNMatrix n){
    ArrayRealVector resRealArray =  this.realArray.ebeMultiply(n.realArray);
    return resRealArray.toArray();
  }

  // Test for the equality of two real vectors.
  public boolean equals(JNMatrix other){
    return this.realArray.equals(other.realArray);
  }
  
  // double[] getDataRef()
  // Get a reference to the underlying data array.

  
  // Returns the size of the vector.

  public int getDimension(){
    return this.realArray.getDimension();
  }

  // double getDistance(RealVector v)
  // Distance between two vectors.


  // Return the entry at the specified index.
  public double getEntry(int index){
    return this.realArray.getEntry(index);
  }

  // double getL1Distance(RealVector v)
  // Distance between two vectors.

  // double getL1Norm()
  // Returns the L1 norm of the vector.

  public double getL1Norm(){
    return this.realArray.getL1Norm();
  }

  // double getLInfDistance(RealVector v)
  // Distance between two vectors.

  // public double getLInfDistance()

  // double getLInfNorm()
  // Returns the L∞ norm of the vector.

  public double getLInfNorm(){
    return this.realArray.getLInfNorm();
  }

  // double getNorm()
  // Returns the L2 norm of the vector.

  public double getNorm(){
    return this.realArray.getNorm();
  }

  // RealVector getSubVector(int index, int n)
  // Get a subvector from consecutive elements.

    // public 

  // int  hashCode()
  // .
  public int hashCode(){
    return this.realArray.hashCode();
  }

  // boolean  isInfinite()
  // Check whether any coordinate of this vector is infinite and none are NaN.

  public boolean isInfinite(){
    return this.realArray.isInfinite();
  }

  // boolean  isNaN()

  public boolean isNaN(){
    return this.realArray.isNaN();
  }

  // Check if any coordinate of this vector is NaN.
  // ArrayRealVector  map(UnivariateFunction function)
  // Acts as if implemented as:

    // public JNMatrix map(UnivariateFunction function){
    //  // JNMatrix res = new JNMatrix(2, new double[]{2,3,4,5}, "FLOAT32", "DENSE_STORE")
    // }


  // RealVector mapAddToSelf(double d)
  // Add a value to each entry.
  public double[] mapAddToSelf(double d){
    RealVector resRealArray =  this.realArray.mapAddToSelf(d);
    return resRealArray.toArray();
  }



  // RealVector mapDivideToSelf(double d)
  // Divide each entry by the argument.
  public double[] mapDivideToSelf(double d){
    RealVector resRealArray =  this.realArray.mapDivideToSelf(d);
    return resRealArray.toArray();
  }


  // RealVector mapMultiplyToSelf(double d)
  // Multiply each entry.
  public double[] mapMultiplyToSelf(double d){
    RealVector resRealArray =  this.realArray.mapMultiplyToSelf(d);
    return resRealArray.toArray();
  }


  // RealVector mapSubtractToSelf(double d)
  // Subtract a value from each entry.
  public double[] mapSubtractToSelf(double d){
    RealVector resRealArray =  this.realArray.mapSubtractToSelf(d);
    return resRealArray.toArray();
  }



  // ArrayRealVector  mapToSelf(UnivariateFunction function)
  // Acts as if it is implemented as:

  // public double[] mapToSelf(Sin){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(Sin);
  //   return resRealArray.toArray();
  // }


  public double[] mapSinToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Sin());
    return resRealArray.toArray();
  }

  public double[] mapCosToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Cos());
    return resRealArray.toArray();
  }

  public double[] mapTanToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Tan());
    return resRealArray.toArray();
  }



  public double[] mapAsinToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Asin());
    return resRealArray.toArray();
  }

  public double[] mapAcosToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Acos());
    return resRealArray.toArray();
  }

  public double[] mapAtanToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Atan());
    return resRealArray.toArray();
  }



  public double[] mapSinhToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Sinh());
    return resRealArray.toArray();
  }

  public double[] mapCoshToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Cosh());
    return resRealArray.toArray();
  }

  public double[] mapTanhToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Tanh());
    return resRealArray.toArray();
  }



  public double[] mapAsinhToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Asinh());
    return resRealArray.toArray();
  }

  public double[] mapAcoshToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Acosh());
    return resRealArray.toArray();
  }

  public double[] mapAtanhToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Atanh());
    return resRealArray.toArray();
  }

  // public double[] mapAtan2ToSelf(){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Atan2());
  //   return resRealArray.toArray();
  // }


  public double[] mapExpToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Exp());
    return resRealArray.toArray();
  }

  public double[] mapLog2ToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Log());
    return resRealArray.toArray();
  }

  public double[] mapLog10ToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Log10());
    return resRealArray.toArray();
  }


  public double[] mapSqrtToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Sqrt());
    return resRealArray.toArray();
  }

  // public double[] mapErfToSelf(double e){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(Erf.erf());
  //   return resRealArray.toArray();
  // }

  // public double[] mapErfcToSelf(double e){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(Erf.erfc());
  //   return resRealArray.toArray();
  // }



  public double[] mapCbrtToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Cbrt());
    return resRealArray.toArray();
  }

  // public double[] mapGammaToSelf(){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Gamma());
  //   return resRealArray.toArray();
  // }

  // public double[] mapAtSelf(){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Sin());
  //   return resRealArray.toArray();
  // }

  public double[] mapFloorToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Floor());
    return resRealArray.toArray();
  }

  public double[] mapCeilToSelf(){
    ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Ceil());
    return resRealArray.toArray();
  }

  // public double[] mapRoundToSelf(){
  //   ArrayRealVector resRealArray =  this.realArray.mapToSelf(new Round());
  //   return resRealArray.toArray();
  // }

  // RealMatrix outerProduct(RealVector v)
  // Compute the outer product.

  public JNMatrix outerProduct(JNMatrix n){
    JNMatrix res= new JNMatrix(this.shape, this.realArray.toArray(), "FLOAT32", "DENSE_STORE");
    return res;
  }


  // void set(double value)
  // Set all elements to a single value.

  // will be used for NMatrix constructors(shortcuts)
  public void set(double value){
    this.realArray.set(value);
  }


  // void setEntry(int index, double value)
  // Set a single element.

  public void setEntry(int index, double value){
    this.realArray.setEntry(index,value);
  }

  // void setSubVector(int index, double[] v)
  // Set a set of consecutive elements.

  public void setSubVector(int index, double[] v){
    this.realArray.setSubVector(index, v);
  }

  // void setSubVector(int index, RealVector v)
  // Set a sequence of consecutive elements.

  public void setSubVector(int index, RealVector v){
    this.setSubVector(index, v);
  }

  // Convert the vector to an array of doubles.
  // double[] toArray()
  public double[] toArray(){
    return this.realArray.toArray();
  }


// String toString()
  public String toString(){
    return this.realArray.toString();
  }

  // double walkInDefaultOrder(RealVectorChangingVisitor visitor)
  // Visits (and possibly alters) all entries of this vector in default order (increasing index).
  // double walkInDefaultOrder(RealVectorChangingVisitor visitor, int start, int end)
  // Visits (and possibly alters) some entries of this vector in default order (increasing index).
  // double walkInDefaultOrder(RealVectorPreservingVisitor visitor)
  // Visits (but does not alter) all entries of this vector in default order (increasing index).
  // double walkInDefaultOrder(RealVectorPreservingVisitor visitor, int start, int end)
  // Visits (but does not alter) some entries of this vector in default order (increasing index).
  // double walkInOptimizedOrder(RealVectorChangingVisitor visitor)
  // Visits (and possibly alters) all entries of this vector in optimized order.
  // double walkInOptimizedOrder(RealVectorChangingVisitor visitor, int start, int end)
  // Visits (and possibly change) some entries of this vector in optimized order.
  // double walkInOptimizedOrder(RealVectorPreservingVisitor visitor)
  // Visits (but does not alter) all entries of this vector in optimized order.
  // double walkInOptimizedOrder(RealVectorPreservingVisitor visitor, int start, int end)
  // Visits (but does not alter) some entries of this vector in optimized order.

}