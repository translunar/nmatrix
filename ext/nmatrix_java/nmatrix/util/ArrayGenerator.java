public class ArrayGenerator{
  // Array from Matrix begin

  public static float[] getArrayFloat(float[][] matrix, int row, int col)
  {
    float[] array = new float[row * col];
    for (int index=0, i=0; i < row ; i++){
        for (int j=0; j < col; j++){
            array[index] = matrix[i][j];
            index++;
        }
    }

    return array;
  }

  public static float[] getArrayColMajor(float[][] matrix, int col, int row)
  {
    float[] array = new float[row * col];
    for (int index=0, i=0; i < col ; i++){
        for (int j=0; j < row; j++){
            array[index] = matrix[i][j];
            index++;
        }
    }

    return array;
  }

  public static float[] getArrayFloatFromDouble(double[][] matrix, int row, int col)
  {
    float[] array = new float[row * col];
    for (int index=0, i=0; i < row ; i++){
        for (int j=0; j < col; j++){
            array[index] = (float)matrix[i][j];
            index++;
        }
    }

    return array;
  }

  

  // Array from Matrix end

  // typeCast beging

  public static float[] convertArrayFloatFromDouble(double[] array){
    float[] resultArray = new float[array.length];
    for (int i=0; i < array.length ; i++){
      array[i] = (float)array[i];
      index++;
    }

    return array;
  }

  // typeCast end
}