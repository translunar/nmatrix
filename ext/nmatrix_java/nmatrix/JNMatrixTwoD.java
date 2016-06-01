import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class JNMatrixTwoD{
	int rows, cols;
	double[] elements;
	RealMatrix nmat2d;

	public void set_rows(int rows){
		this.rows = rows;
	}

	public int get_rows(){
		return rows;
	}

	public void set_cols(int cols){
		this.cols = cols;
	}

	public int get_cols(){
		return cols;
	}

	public double[][] two_d_array_generator(int[] shape, double[] oneDArray){
		double[][] twoDArray = new double[shape[0]][shape[1]];
		for (int i=0,index=0; i < shape[0];i++ ){
			for(int j=0; j< shape[1]; j++){
				twoDArray[i][j] = oneDArray[index];
				index++;
			}
		}
		return twoDArray;
	}
	

	public JNMatrixTwoD(int[] shape, double[] oneDArray){
		set_rows(shape[0]);
		set_cols(shape[1]);
		nmat2d = MatrixUtils.createRealMatrix(two_d_array_generator(shape, oneDArray));
	}


	// Methods derived from Array2dRealMatrix

	// Array2DRowRealMatrix	add(Array2DRowRealMatrix m)
	// Compute the sum of this and m.
	// void	addToEntry(int row, int column, double increment)
	// Adds (in place) the specified value to the specified entry of this matrix.
	// RealMatrix	copy()
	// Returns a (deep) copy of this.
	// RealMatrix	createMatrix(int rowDimension, int columnDimension)
	// Create a new RealMatrix of the same type as the instance with the supplied row and column dimensions.
	// int	getColumnDimension()
	// Returns the number of columns of this matrix.
	// double[][]	getData()
	// Returns matrix entries as a two-dimensional array.
	// double[][]	getDataRef()
	// Get a reference to the underlying data array.
	// double	getEntry(int row, int column)
	// Get the entry in the specified row and column.
	// int	getRowDimension()
	// Returns the number of rows of this matrix.
	// Array2DRowRealMatrix	multiply(Array2DRowRealMatrix m)
	// Returns the result of postmultiplying this by m.
	// void	multiplyEntry(int row, int column, double factor)
	// Multiplies (in place) the specified entry of this matrix by the specified value.
	// double[]	operate(double[] v)
	// Returns the result of multiplying this by the vector v.
	// double[]	preMultiply(double[] v)
	// Returns the (row) vector result of premultiplying this by the vector v.
	// void	setEntry(int row, int column, double value)
	// Set the entry in the specified row and column.
	// void	setSubMatrix(double[][] subMatrix, int row, int column)
	// Replace the submatrix starting at row, column using data in the input subMatrix array.
	// Array2DRowRealMatrix	subtract(Array2DRowRealMatrix m)
	// Returns this minus m.
	// double	walkInColumnOrder(RealMatrixChangingVisitor visitor)
	// Visit (and possibly change) all matrix entries in column order.
	// double	walkInColumnOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn)
	// Visit (and possibly change) some matrix entries in column order.
	// double	walkInColumnOrder(RealMatrixPreservingVisitor visitor)
	// Visit (but don't change) all matrix entries in column order.
	// double	walkInColumnOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn)
	// Visit (but don't change) some matrix entries in column order.
	// double	walkInRowOrder(RealMatrixChangingVisitor visitor)
	// Visit (and possibly change) all matrix entries in row order.
	// double	walkInRowOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn)
	// Visit (and possibly change) some matrix entries in row order.
	// double	walkInRowOrder(RealMatrixPreservingVisitor visitor)
	// Visit (but don't change) all matrix entries in row order.
	// double	walkInRowOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn)
	// Visit (but don't change) some matrix entries in row order.
}