import org.apache.commons.math3.util.FastMath;

public class MathHelper{

  public static double[] log(double base, double[] arr){
    double[] result = new double[arr.length];
    for(int i = 0; i< arr.length; i++){
      result[i] = FastMath.log(base, arr[i]);
    } 
    return result;
  }

}