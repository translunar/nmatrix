require 'spec_helper'
require 'lapack_shared'
require "./lib/nmatrix/atlas"

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
end
