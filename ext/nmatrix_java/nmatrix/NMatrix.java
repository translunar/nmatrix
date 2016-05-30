import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.ArrayFieldVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.FieldVector;

public class NMatrix{

	private int shape;
	public ArrayRealVector realArray;
	private String dtype_string;
	private String stype_string;

	private Dense_Storage storage;
	
	public String get_stype_string(){
		return this.stype_string;
	}


  private Stype stype;
  private Dtype dtype;

	// NMatrix(shape, elements, dtype, stype)
	public NMatrix(int shape, double[] elements, Dtype dtype, Stype stype){
		this.shape = shape;
		this.realArray = new ArrayRealVector(elements);
		this.dtype = dtype;

		// if(this.checkVectorDimensions(2)){
		// 	//we have a double matrix
		// }

		switch(stype){
			case DENSE_STORE:
				this.stype_string = "dense";
				// (int dim, int[] shape, String dtype, int offset, int count, double[] elements)
				// this.storage = new Dense_Storage( 2,new int[]{2,3}, "int", 1, 4, elements);
				break;
			case LIST_STORE:
				this.stype_string = "list";
				break;
			case YALE_STORE:
				this.stype_string = "yale";
				break;
			default:
				this.stype_string ="stype could not be determined";
				break;
		}

		switch(dtype){
			case BYTE:
				this.dtype_string = "BYTE";
				break;
			case INT8:
				this.dtype_string = "INT8";
				break;
			case INT16:
				this.dtype_string = "INT16";
				break;
			case INT32:
				this.dtype_string = "INT32";
				break;
			case INT64:
				this.dtype_string = "INT64";
				break;
			case FLOAT32:
				this.dtype_string = "FLOAT32";
				break;
			case FLOAT64:
				this.dtype_string = "FLOAT64";
				break;
			case COMPLEX64:
				this.dtype_string = "COMPLEX64";
				break;
			case COMPLEX128:
				this.dtype_string = "COMPLEX128";
				break;
			case RUBYOBJ:
				this.dtype_string = "RUBYOBJ";
				break;
		}
	}


}