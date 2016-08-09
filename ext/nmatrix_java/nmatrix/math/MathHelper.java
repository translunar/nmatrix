import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.special.Gamma;

public class MathHelper{

  public static double[] log(double base, double[] arr){
    double[] result = new double[arr.length];
    for(int i = 0; i< arr.length; i++){
      result[i] = FastMath.log(base, arr[i]);
    } 
    return result;
  }

  public static double[] erf(double[] arr){
    double[] result = new double[arr.length];
    for(int i = 0; i< arr.length; i++){
      result[i] = Erf.erf(arr[i]);
    } 
    return result;
  }

  public static double[] erfc(double[] arr){
    double[] result = new double[arr.length];
    for(int i = 0; i< arr.length; i++){
      result[i] = Erf.erfc(arr[i]);
    } 
    return result;
  }

  public static double[] gamma(double[] arr){
    double[] result = new double[arr.length];
    for(int i = 0; i< arr.length; i++){
      result[i] = Gamma.gamma(arr[i]);
    } 
    return result;
  }

  public static double[] round(double[] arr){
    double[] result = new double[arr.length];
    for(int i = 0; i< arr.length; i++){
      result[i] = Math.round(arr[i]);
    } 
    return result;
  }

  public static double[] ldexp(double[] arr1, double[] arr2){
    double[] result = new double[arr1.length];
    for(int i = 0; i< arr1.length; i++){
      result[i] = arr1[i] * Math.pow(2, arr2[i]);
    } 
    return result;
  }

  public static double[] hypot(double[] arr1, double[] arr2){
    double[] result = new double[arr1.length];
    for(int i = 0; i< arr1.length; i++){
      result[i] =  Math.sqrt(arr2[i] * arr2[i] + arr1[i] * arr1[i]);
    } 
    return result;
  }

}