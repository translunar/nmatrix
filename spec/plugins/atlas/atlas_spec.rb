require 'spec_helper'
require 'lapack_shared'
require "./lib/nmatrix/atlas"
require 'blas_shared'
require 'math_shared'

describe "NMatrix::LAPACK implementation from nmatrix-atlas plugin" do
  it "returns 3" do
    n = NMatrix.new([2,2], [0,1,2,3], dtype: :int64)
    expect(n.test_return_3).to eq(3)
  end
  it "returns 2" do
    n = NMatrix.new([2,2], [0,1,2,3], dtype: :int64)
    expect(n.test_c_ext_return_2).to eq(2)
  end

  include_examples "LAPACK shared"
  include_examples "math shared"
  include_examples "BLAS shared"

  #gesvd, gesdd don't have to be shared
  [:float32, :float64, :complex64, :complex128].each do |dtype|
    context dtype do
      it "exposes lapack_gesvd" do
        if [:float32, :float64].include? dtype
          # http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.htm
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([6,5], %w|8.79 9.93 9.83 5.45 3.16
                                    6.11 6.91 5.04 -0.27 7.98
                                    -9.15 -7.93 4.86 4.85 3.01
                                    9.57 1.64 8.83 0.74 5.80
                                    -3.49 4.02 9.80 10.00 4.27
                                    9.84 0.15 -8.99 -6.02 -5.31|.map(&:to_f), dtype: dtype)
          s_true = NMatrix.new([5,1], [27.468732418221848, 22.643185009774697, 8.55838822848258, 5.985723201512129, 2.014899658715758], dtype: dtype)
          u_true = NMatrix.new([6,6], [0.5911423764124366, 0.2631678147140566, 0.35543017386282705, 0.3142643627269277, 0.22993831536474876, 0.5507531798028814, 0.39756679420242563, 0.24379902792633035, -0.2223900006854459, -0.7534661509534583, -0.36358968669749675, 0.18203479013503598, 0.03347896906244713, -0.6002725806935827, -0.45083926892230775, 0.2334496572447145, -0.30547573274793166, 0.5361732698764649, 0.42970690313701826, 0.23616680628112557, -0.6858628638738115, 0.3318600182003095, 0.16492763488450998, -0.3896628703606129, 0.46974792156665846, -0.350891398883702, 0.3874446030996732, 0.15873555958215635, -0.5182574373535355, -0.46077222860548506, -0.2933587584644035, 0.5762621191338902, -0.020852917980871157, 0.37907766706016066, -0.6525516005923975, 0.1091068082007298], dtype: dtype)
          vt_true = NMatrix.new([5,5], [0.2513827927204964, 0.3968455517769292, 0.6921510074703637, 0.3661704447722308, 0.4076352386533526, 0.814836686086339, 0.3586615001880026, -0.24888801115928444, -0.36859353794461763, -0.0979625692668867, -0.26061850558422095, 0.7007682094072527, -0.22081144672043734, 0.385938483188542, -0.4932501428510235, 0.3967237771305969, -0.4507112412166428, 0.25132114969375313, 0.43424860143667116, -0.6226840720358043, -0.21802776368654578, 0.1402099498711204, 0.5891194492399431, -0.6265282503648171, -0.4395516923423329], dtype: dtype)
          s   = NMatrix.new([5,1], 0, dtype: dtype)
          u   = NMatrix.new([6,6], 0, dtype: dtype)
          ldu = 6
          vt  = NMatrix.new([5,5], 0, dtype: dtype)
          ldvt= 5
        elsif [:complex64, :complex128].include? dtype
          #http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/cgesvd_ex.c.htm
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([3,4], [[  5.91, -5.69], [  7.09,  2.72], [  7.78, -4.06], [ -0.79, -7.21], [ -3.15, -4.08], [ -1.89,  3.27], [  4.57, -2.07], [ -3.88, -3.30], [ -4.89,  4.20], [  4.10, -6.70], [  3.28, -3.84], [  3.84,  1.19]].map {|e| Complex(*e) } , dtype: dtype)
          s_true = NMatrix.new([3,1], [17.625362929191322, 11.610180169432551, 6.782853237952547], dtype: dtype)
          u_true = NMatrix.new([3,3], [[-0.8567519102348325, 0.0], [0.40169740961685524, 0.0], [-0.32344297088679147, 0.0], [-0.35054713697315854, 0.12848536699843283], [-0.24033104987380846, -0.21020102698539941], [0.6300695559577525, -0.6013959466135734], [0.15049130902203472, 0.32239222299087716], [0.6074132024425246, 0.6064197451653939], [0.35574306405361666, -0.1008304865288007]].map {|e| Complex(*e) }, dtype: dtype)
          vt_true = NMatrix.new([4,4], [[-0.21930077612929075, 0.5060004210000614], [-0.37075567509741925, -0.31567672661135066], [-0.5263931112934799, 0.11242515072234287], [0.14606726521281474, 0.384310303136579], [0.3070931721780084, 0.3057047470480451], [0.08977335255770591, -0.572474744963393], [0.18308695699042435, -0.3871008815037038], [0.3757834107155146, -0.3897059208347286], [-0.531582078688174, -0.23937282275405594], [-0.48895294222998703, -0.2839733131101101], [0.4661677577500858, 0.25387318970802736], [0.15355216221960952, -0.18725138645583775], [-0.15490184749754707, -0.3797837513071778], [-0.10357983419871576, 0.3108974764804632], [-0.280822158137883, -0.4077718630158068], [0.6911732564070943, 0.03904164199100488]].map {|e| Complex(*e) }, dtype: dtype)
          s   = NMatrix.new([3,1], 0, dtype: dtype)
          u   = NMatrix.new([3,3], 0, dtype: dtype)
          ldu = 3
          vt  = NMatrix.new([4,4], 0, dtype: dtype)
          ldvt= 4
        else 
          a = NMatrix.new([4,3], dtype: dtype)
        end
        err = case dtype
              when :float32, :complex64
                1e-6
              when :float64, :complex128
                1e-15
              else
                1e-64 # FIXME: should be 0, but be_within(0) does not work.
              end
        err = err *5e1
        a_clone = a.clone #clone a so we can check our results later
        begin
          # There is a subtlety here. The LAPACK *gesvd functions expect a matrix stored in column-major form, so we need to some adjusting for this. See lib/nmatrix/lapack.rb for an explanation of why we use these arguments
          info = NMatrix::LAPACK::lapack_gesvd(:a, :a, a.shape[1], a.shape[0], a, a.shape[1], s, vt, ldvt, u, ldu, 500)
        rescue NotImplementedError => e
          pending e.to_s
        end

        expect(u).to be_within(err).of(u_true)
        expect(vt).to be_within(err).of(vt_true)
        expect(s).to be_within(err).of(s_true)

        #construct sigma to check decomposition
        sigma = NMatrix.zeros(a.shape, dtype: dtype)
        s.to_a.flatten.each_with_index {|x, i| sigma[i,i] = x}
        expect(u.dot(sigma).dot(vt)).to be_within(err).of(a_clone)

        #check unitarity of u and vt
        expect(u.dot(u.conjugate_transpose)).to be_within(err).of(NMatrix.eye(u.shape,dtype: dtype))
        expect(vt.dot(vt.conjugate_transpose)).to be_within(err).of(NMatrix.eye(vt.shape,dtype: dtype))
      end

      it "exposes lapack_gesdd" do
        if [:float32, :float64].include? dtype
          # http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.htm
          # we use the same example for both gesvd and gesdd
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([6,5], %w|8.79 9.93 9.83 5.45 3.16
                                    6.11 6.91 5.04 -0.27 7.98
                                    -9.15 -7.93 4.86 4.85 3.01
                                    9.57 1.64 8.83 0.74 5.80
                                    -3.49 4.02 9.80 10.00 4.27
                                    9.84 0.15 -8.99 -6.02 -5.31|.map(&:to_f), dtype: dtype)
          s_true = NMatrix.new([5,1], [27.468732418221848, 22.643185009774697, 8.55838822848258, 5.985723201512129, 2.014899658715758], dtype: dtype)
          u_true = NMatrix.new([6,6], [0.5911423764124366, 0.2631678147140566, 0.35543017386282705, 0.3142643627269277, 0.22993831536474876, 0.5507531798028814, 0.39756679420242563, 0.24379902792633035, -0.2223900006854459, -0.7534661509534583, -0.36358968669749675, 0.18203479013503598, 0.03347896906244713, -0.6002725806935827, -0.45083926892230775, 0.2334496572447145, -0.30547573274793166, 0.5361732698764649, 0.42970690313701826, 0.23616680628112557, -0.6858628638738115, 0.3318600182003095, 0.16492763488450998, -0.3896628703606129, 0.46974792156665846, -0.350891398883702, 0.3874446030996732, 0.15873555958215635, -0.5182574373535355, -0.46077222860548506, -0.2933587584644035, 0.5762621191338902, -0.020852917980871157, 0.37907766706016066, -0.6525516005923975, 0.1091068082007298], dtype: dtype)
          vt_true = NMatrix.new([5,5], [0.2513827927204964, 0.3968455517769292, 0.6921510074703637, 0.3661704447722308, 0.4076352386533526, 0.814836686086339, 0.3586615001880026, -0.24888801115928444, -0.36859353794461763, -0.0979625692668867, -0.26061850558422095, 0.7007682094072527, -0.22081144672043734, 0.385938483188542, -0.4932501428510235, 0.3967237771305969, -0.4507112412166428, 0.25132114969375313, 0.43424860143667116, -0.6226840720358043, -0.21802776368654578, 0.1402099498711204, 0.5891194492399431, -0.6265282503648171, -0.4395516923423329], dtype: dtype)
          s   = NMatrix.new([5,1], 0, dtype: dtype)
          u   = NMatrix.new([6,6], 0, dtype: dtype)
          ldu = 6
          vt  = NMatrix.new([5,5], 0, dtype: dtype)
          ldvt= 5
        elsif [:complex64, :complex128].include? dtype
          #http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/cgesvd_ex.c.htm
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([3,4], [[  5.91, -5.69], [  7.09,  2.72], [  7.78, -4.06], [ -0.79, -7.21], [ -3.15, -4.08], [ -1.89,  3.27], [  4.57, -2.07], [ -3.88, -3.30], [ -4.89,  4.20], [  4.10, -6.70], [  3.28, -3.84], [  3.84,  1.19]].map {|e| Complex(*e) } , dtype: dtype)
          s_true = NMatrix.new([3,1], [17.625362929191322, 11.610180169432551, 6.782853237952547], dtype: dtype)
          u_true = NMatrix.new([3,3], [[-0.8567519102348325, 0.0], [-0.40169740961685496, 0.0], [0.3234429708867915, 0.0], [-0.35054713697315854, 0.12848536699843277], [0.2403310498738087, 0.21020102698539955], [-0.6300695559577525, 0.6013959466135734], [0.1504913090220347, 0.322392222990877], [-0.6074132024425248, -0.6064197451653935], [-0.3557430640536165, 0.10083048652880122]].map {|e| Complex(*e) }, dtype: dtype)
          vt_true = NMatrix.new([4,4], [[-0.21930077612929055, 0.5060004210000613], [-0.37075567509741914, -0.3156767266113506], [-0.5263931112934799, 0.11242515072234296], [0.1460672652128147, 0.384310303136579], [-0.3070931721780085, -0.30570474704804507], [-0.08977335255770649, 0.5724747449633932], [-0.18308695699042432, 0.38710088150370364], [-0.37578341071551463, 0.3897059208347283], [0.531582078688174, 0.23937282275405627], [0.4889529422299872, 0.2839733131101101], [-0.46616775775008557, -0.25387318970802764], [-0.15355216221960918, 0.18725138645583758], [-0.14694036331555857, -0.3829344718666594], [-0.11004704214257588, 0.3086675743420804], [-0.2722490037071734, -0.4135449875757793], [0.6902076866535576, 0.05346091531282677]].map {|e| Complex(*e) }, dtype: dtype)
          s   = NMatrix.new([3,1], 0, dtype: dtype)
          u   = NMatrix.new([3,3], 0, dtype: dtype)
          ldu = 3
          vt  = NMatrix.new([4,4], 0, dtype: dtype)
          ldvt= 4
        else 
          a = NMatrix.new([4,3], dtype: dtype)
        end
        err = case dtype
              when :float32, :complex64
                1e-6
              when :float64, :complex128
                1e-15
              else
                1e-64 # FIXME: should be 0, but be_within(0) does not work.
              end
        err = err *5e1
        a_clone = a.clone #clone a so we can compare to it later
        begin
          # There is a subtlety here. The LAPACK *gesdd functions expect a matrix stored in column-major form, so we need to some adjusting for this. See lib/nmatrix/lapack.rb for an explanation of why we use these arguments
          info = NMatrix::LAPACK::lapack_gesdd(:a, a.shape[1], a.shape[0], a, a.shape[1], s, vt, ldvt, u, ldu, 500)
        rescue NotImplementedError => e
          pending e.to_s
        end

        expect(u).to be_within(err).of(u_true)
        expect(vt).to be_within(err).of(vt_true)
        expect(s).to be_within(err).of(s_true)

        #construct sigma to check decomposition
        sigma = NMatrix.zeros(a.shape, dtype: dtype)
        s.to_a.flatten.each_with_index {|x, i| sigma[i,i] = x}
        expect(u.dot(sigma).dot(vt)).to be_within(err).of(a_clone)

        #check unitarity of u and vt
        expect(u.dot(u.conjugate_transpose)).to be_within(err).of(NMatrix.eye(u.shape,dtype: dtype))
        expect(vt.dot(vt.conjugate_transpose)).to be_within(err).of(NMatrix.eye(vt.shape,dtype: dtype))
      end
 
      it "exposes the convenience gesvd method" do
        if [:float32, :float64].include? dtype
          # http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.htm
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([6,5], %w|8.79 9.93 9.83 5.45 3.16
                                    6.11 6.91 5.04 -0.27 7.98
                                    -9.15 -7.93 4.86 4.85 3.01
                                    9.57 1.64 8.83 0.74 5.80
                                    -3.49 4.02 9.80 10.00 4.27
                                    9.84 0.15 -8.99 -6.02 -5.31|.map(&:to_f), dtype: dtype)
          s_true = NMatrix.new([5,1], [27.468732418221848, 22.643185009774697, 8.55838822848258, 5.985723201512129, 2.014899658715758], dtype: dtype)
          u_true = NMatrix.new([6,6], [0.5911423764124366, 0.2631678147140566, 0.35543017386282705, 0.3142643627269277, 0.22993831536474876, 0.5507531798028814, 0.39756679420242563, 0.24379902792633035, -0.2223900006854459, -0.7534661509534583, -0.36358968669749675, 0.18203479013503598, 0.03347896906244713, -0.6002725806935827, -0.45083926892230775, 0.2334496572447145, -0.30547573274793166, 0.5361732698764649, 0.42970690313701826, 0.23616680628112557, -0.6858628638738115, 0.3318600182003095, 0.16492763488450998, -0.3896628703606129, 0.46974792156665846, -0.350891398883702, 0.3874446030996732, 0.15873555958215635, -0.5182574373535355, -0.46077222860548506, -0.2933587584644035, 0.5762621191338902, -0.020852917980871157, 0.37907766706016066, -0.6525516005923975, 0.1091068082007298], dtype: dtype)
          vt_true = NMatrix.new([5,5], [0.2513827927204964, 0.3968455517769292, 0.6921510074703637, 0.3661704447722308, 0.4076352386533526, 0.814836686086339, 0.3586615001880026, -0.24888801115928444, -0.36859353794461763, -0.0979625692668867, -0.26061850558422095, 0.7007682094072527, -0.22081144672043734, 0.385938483188542, -0.4932501428510235, 0.3967237771305969, -0.4507112412166428, 0.25132114969375313, 0.43424860143667116, -0.6226840720358043, -0.21802776368654578, 0.1402099498711204, 0.5891194492399431, -0.6265282503648171, -0.4395516923423329], dtype: dtype)
        elsif [:complex64, :complex128].include? dtype
          #http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/cgesvd_ex.c.htm
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([3,4], [[  5.91, -5.69], [  7.09,  2.72], [  7.78, -4.06], [ -0.79, -7.21], [ -3.15, -4.08], [ -1.89,  3.27], [  4.57, -2.07], [ -3.88, -3.30], [ -4.89,  4.20], [  4.10, -6.70], [  3.28, -3.84], [  3.84,  1.19]].map {|e| Complex(*e) } , dtype: dtype)
          s_true = NMatrix.new([3,1], [17.625362929191322, 11.610180169432551, 6.782853237952547], dtype: dtype)
          u_true = NMatrix.new([3,3], [[-0.8567519102348325, 0.0], [0.40169740961685524, 0.0], [-0.32344297088679147, 0.0], [-0.35054713697315854, 0.12848536699843283], [-0.24033104987380846, -0.21020102698539941], [0.6300695559577525, -0.6013959466135734], [0.15049130902203472, 0.32239222299087716], [0.6074132024425246, 0.6064197451653939], [0.35574306405361666, -0.1008304865288007]].map {|e| Complex(*e) }, dtype: dtype)
          vt_true = NMatrix.new([4,4], [[-0.21930077612929075, 0.5060004210000614], [-0.37075567509741925, -0.31567672661135066], [-0.5263931112934799, 0.11242515072234287], [0.14606726521281474, 0.384310303136579], [0.3070931721780084, 0.3057047470480451], [0.08977335255770591, -0.572474744963393], [0.18308695699042435, -0.3871008815037038], [0.3757834107155146, -0.3897059208347286], [-0.531582078688174, -0.23937282275405594], [-0.48895294222998703, -0.2839733131101101], [0.4661677577500858, 0.25387318970802736], [0.15355216221960952, -0.18725138645583775], [-0.15490184749754707, -0.3797837513071778], [-0.10357983419871576, 0.3108974764804632], [-0.280822158137883, -0.4077718630158068], [0.6911732564070943, 0.03904164199100488]].map {|e| Complex(*e) }, dtype: dtype)
        else 
          a = NMatrix.new([4,3], dtype: dtype)
        end
        err = case dtype
              when :float32, :complex64
                1e-6
              when :float64, :complex128
                1e-15
              else
                1e-64 # FIXME: should be 0, but be_within(0) does not work.
              end
        err = err *5e1
        begin
          u, s, vt = a.gesvd
        rescue NotImplementedError => e
          pending e.to_s
        end
        expect(u).to be_within(err).of(u_true)
        expect(vt).to be_within(err).of(vt_true)
        expect(s).to be_within(err).of(s_true)

        #construct sigma to check decomposition
        sigma = NMatrix.zeros(a.shape, dtype: dtype)
        s.to_a.flatten.each_with_index {|x, i| sigma[i,i] = x}
        expect(u.dot(sigma).dot(vt)).to be_within(err).of(a)

        #check unitarity of u and vt
        expect(u.dot(u.conjugate_transpose)).to be_within(err).of(NMatrix.eye(u.shape,dtype: dtype))
        expect(vt.dot(vt.conjugate_transpose)).to be_within(err).of(NMatrix.eye(vt.shape,dtype: dtype))
      end

      it "exposes the convenience gesdd method" do
        if [:float32, :float64].include? dtype
          # http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.htm
          # we use the same example for gesdd and gesvd
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([6,5], %w|8.79 9.93 9.83 5.45 3.16
                                    6.11 6.91 5.04 -0.27 7.98
                                    -9.15 -7.93 4.86 4.85 3.01
                                    9.57 1.64 8.83 0.74 5.80
                                    -3.49 4.02 9.80 10.00 4.27
                                    9.84 0.15 -8.99 -6.02 -5.31|.map(&:to_f), dtype: dtype)
          s_true = NMatrix.new([5,1], [27.468732418221848, 22.643185009774697, 8.55838822848258, 5.985723201512129, 2.014899658715758], dtype: dtype)
          u_true = NMatrix.new([6,6], [0.5911423764124366, 0.2631678147140566, 0.35543017386282705, 0.3142643627269277, 0.22993831536474876, 0.5507531798028814, 0.39756679420242563, 0.24379902792633035, -0.2223900006854459, -0.7534661509534583, -0.36358968669749675, 0.18203479013503598, 0.03347896906244713, -0.6002725806935827, -0.45083926892230775, 0.2334496572447145, -0.30547573274793166, 0.5361732698764649, 0.42970690313701826, 0.23616680628112557, -0.6858628638738115, 0.3318600182003095, 0.16492763488450998, -0.3896628703606129, 0.46974792156665846, -0.350891398883702, 0.3874446030996732, 0.15873555958215635, -0.5182574373535355, -0.46077222860548506, -0.2933587584644035, 0.5762621191338902, -0.020852917980871157, 0.37907766706016066, -0.6525516005923975, 0.1091068082007298], dtype: dtype)
          vt_true = NMatrix.new([5,5], [0.2513827927204964, 0.3968455517769292, 0.6921510074703637, 0.3661704447722308, 0.4076352386533526, 0.814836686086339, 0.3586615001880026, -0.24888801115928444, -0.36859353794461763, -0.0979625692668867, -0.26061850558422095, 0.7007682094072527, -0.22081144672043734, 0.385938483188542, -0.4932501428510235, 0.3967237771305969, -0.4507112412166428, 0.25132114969375313, 0.43424860143667116, -0.6226840720358043, -0.21802776368654578, 0.1402099498711204, 0.5891194492399431, -0.6265282503648171, -0.4395516923423329], dtype: dtype)
        elsif [:complex64, :complex128].include? dtype
          #http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/cgesvd_ex.c.htm
          # our result does not exactly match the result in the intel docs, but it is still a valid singular value decomposition
          a = NMatrix.new([3,4], [[  5.91, -5.69], [  7.09,  2.72], [  7.78, -4.06], [ -0.79, -7.21], [ -3.15, -4.08], [ -1.89,  3.27], [  4.57, -2.07], [ -3.88, -3.30], [ -4.89,  4.20], [  4.10, -6.70], [  3.28, -3.84], [  3.84,  1.19]].map {|e| Complex(*e) } , dtype: dtype)
          s_true = NMatrix.new([3,1], [17.625362929191322, 11.610180169432551, 6.782853237952547], dtype: dtype)
          u_true = NMatrix.new([3,3], [[-0.8567519102348325, 0.0], [-0.40169740961685496, 0.0], [0.3234429708867915, 0.0], [-0.35054713697315854, 0.12848536699843277], [0.2403310498738087, 0.21020102698539955], [-0.6300695559577525, 0.6013959466135734], [0.1504913090220347, 0.322392222990877], [-0.6074132024425248, -0.6064197451653935], [-0.3557430640536165, 0.10083048652880122]].map {|e| Complex(*e) }, dtype: dtype)
          vt_true = NMatrix.new([4,4], [[-0.21930077612929055, 0.5060004210000613], [-0.37075567509741914, -0.3156767266113506], [-0.5263931112934799, 0.11242515072234296], [0.1460672652128147, 0.384310303136579], [-0.3070931721780085, -0.30570474704804507], [-0.08977335255770649, 0.5724747449633932], [-0.18308695699042432, 0.38710088150370364], [-0.37578341071551463, 0.3897059208347283], [0.531582078688174, 0.23937282275405627], [0.4889529422299872, 0.2839733131101101], [-0.46616775775008557, -0.25387318970802764], [-0.15355216221960918, 0.18725138645583758], [-0.14694036331555857, -0.3829344718666594], [-0.11004704214257588, 0.3086675743420804], [-0.2722490037071734, -0.4135449875757793], [0.6902076866535576, 0.05346091531282677]].map {|e| Complex(*e) }, dtype: dtype)
        else 
          a = NMatrix.new([4,3], dtype: dtype)
        end
        err = case dtype
              when :float32, :complex64
                1e-6
              when :float64, :complex128
                1e-15
              else
                1e-64 # FIXME: should be 0, but be_within(0) does not work.
              end
        err = err *5e1
        begin
          u, s, vt = a.gesdd
        rescue NotImplementedError => e
          pending e.to_s
        end
        expect(u).to be_within(err).of(u_true)
        expect(vt).to be_within(err).of(vt_true)
        expect(s).to be_within(err).of(s_true)

        #construct sigma to check decomposition
        sigma = NMatrix.zeros(a.shape, dtype: dtype)
        s.to_a.flatten.each_with_index {|x, i| sigma[i,i] = x}
        expect(u.dot(sigma).dot(vt)).to be_within(err).of(a)

        #check unitarity of u and vt
        expect(u.dot(u.conjugate_transpose)).to be_within(err).of(NMatrix.eye(u.shape,dtype: dtype))
        expect(vt.dot(vt.conjugate_transpose)).to be_within(err).of(NMatrix.eye(vt.shape,dtype: dtype))
      end
    end
  end
end
