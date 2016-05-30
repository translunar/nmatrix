// public class Common{
// 	public static  int storage_count_max_elements(STORAGE storage) {
//     int i;
//     int count = 1;

//     for (i = storage.dim; i-- > 0;) {
//       count *= storage.shape[i];
//     }

//     return count;
//   }

//   // Helper function used only for the RETURN_SIZED_ENUMERATOR macro. Returns the length of
//   // the matrix's storage.
//   public static long nm_enumerator_length(NMatrix nmatrix) {
//     long len = storage_count_max_elements(getDenseStorage(nmatrix));
//     return len;
//   }
// }