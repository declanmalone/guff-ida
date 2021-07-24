
//! # A crate for working with Cauchy and Vandermonde matrices
//!
//! A small collection of routines for creating matrices that can be
//! used to implement erasure (error-correction) schemes or threshold
//! schemes using Galois fields.
//!
//! Note that this crate only provides data for insertion into a
//! matrix. For the missing functionality, see:
//!
//! * [guff](https://crates.io/crates/guff) : basic operations over
//!   finite fields, including vector operations
//! * [guff-matrix](https://crates.io/crates/guff-matrix) : full set
//!   of matrix types and operations
//!
//! # Using finite field matrix operations for threshold schemes
//! 
//! A "threshold scheme" is a mathematical method for securely
//! splitting a secret into a number of "shares" such that:
//!
//! * if a set number (the "threshold") of shares are combined, the
//! original secret can be recovered
//!
//! * if fewer shares than the threshold are combined, no information
//! about the secret is revealed
//!
//! ## Module Focus
//!
//! This module focuses in particular on Michael O. Rabin's
//! "Information Dispersal Algorithm" (IDA). In it, splitting a secret
//! is achieved by:
//! 
//! * creating a transform matrix that has the required threshold property
//!
//! * placing the secret into an input matrix, padding it if needed
//!
//! * calculating transform x input to get an output matrix
//!
//! * reading off each share as a row of the transform matrix and the
//!   corresponding row of the output matrix
//!
//! To reconstruct the secret:
//!
//! * take the supplied transform matrix rows and put them into a
//!   square matrix
//!
//! * calculate the inverse of the matrix
//!
//! * form a new input matrix from the corresponding output data rows
//!
//! * calculate inverse x input
//!
//! * read the secret back from the output matrix
//!
//! More details of the algorithm can be found [later](todo).
//!

// will need to use external gf_2px
//use num;
use num_traits::identities::{One,Zero};
use guff::*;

//impl From<u32> for NumericOps {
//     fn from(val: u32) -> Self { val as Self }
//}


/// ## Vandermonde-form matrix
/// ```ascii
///
///     |     0    1    2          k-1  |
///     |    0    0    0   ...    0     |
///     |                               |
///     |     0    1    2          k-1  |
///     |    1    1    1   ...    1     |
///     |                               |
///     |    :    :    :    :     :     |
///     |                               |
///     |     0    1    2          k-1  |
///     |  n-1  n-1  n-1   ...  n-1     |
/// ```
///
/// Can be used for Reed-Solomon coding, or a version of it,
/// anyway. This is not the most general form of a Vandermonde matrix,
/// but it is useful as a particular case since it doesn't require any
/// parameters to produce it.
///
/// Return is as a single vector of n rows, each of k elements

pub fn vandermonde_matrix<G> (field : &G, k: usize, n : usize)
    -> Vec<G::E>
where G : GaloisField, G::E : Into<usize>, G::EE : Into<usize>
{
    let zero = G::E::zero();
    let one  = G::E::one();

    let mut v = Vec::<G::E>::new();
    if k < 1 || n < 1 {
	return v
    }

    // If pow() is expensive, can use repeated multiplications below.
    
    // range op won't work, so use while loop
    
    let mut row = zero;
    while row.into() < n {
	let mut col = G::EE::zero();
	while col.into() < k {
	    v.push(field.pow(row, col));
	    col = col + G::EE::one();
	}
	row = row + one
    }
    v
}

/// ## Cauchy-form matrix generated from a key
///
/// ```ascii
///                 k columns
///
///     |     1        1             1     |
///     |  -------  -------  ...  -------  |
///     |  x1 + y1  x1 + y2       x1 + yk  |
///     |                                  |
///     |     1        1             1     |
///     |  -------  -------  ...  -------  |
///     |  x2 + y1  x2 + y2       x2 + yk  |  n rows
///     |                                  |
///     |     :        :      :      :     |
///     |                                  |
///     |     1        1             1     |
///     |  -------  -------  ...  -------  |
///     |  xn + y1  xn + y2       xn + yk  |
///```
///
/// All [y1 .. yk, x1 .. xn] field values must be distinct non-zero
/// values
///
/// **TODO**: Check that all input values are distinct. Can put all
/// elements on a heap and then check that no parent node has a child
/// node equal to it. Alternatively, check the condition as we're
/// building the heap? That should work, too.
///
/// **TODO**: use a random number number generator to select k + n distinct
/// field elements (eg, Floyd's algorithm for shuffling/selection from
/// an array of all field elements if the field size is small, or a
/// modification of the heap approach above for when it's impractical
/// to list all the field elements)
///
/// I've ordered the elements with y values first since these will be
/// reused across all rows. I will allow passing in a vector that has
/// more x values than are required
///
/// We don't operate on k, n to produce field values, so they can be
/// passed in as regular types
pub fn cauchy_matrix<G>
    (field : &G, key : &Vec<G::E>, k: usize, n : usize)
    -> Vec<G::E>
where G : GaloisField,
{
    let mut v = Vec::<G::E>::with_capacity(k * n);
    let vlen = key.len();

    // I can change signature to return a Result later, but for now
    // can return a null vector to indicate failure.
    let min_key_size = k + n;
    if vlen < min_key_size {
	println!("Key should have at least {} elements", min_key_size);
	return v
    }

    // slice format?
    let y : &[G::E] = &key[..k];
    let x : &[G::E] = &key[k..];

    // populate vector row by row
    for i in 0..n {
	for j in 0..k {
	    v.push(field.inv(x[i] ^ y[j]))
	}
    }
    v
}

/// # Generate inverse Cauchy matrix using a key
///
/// If the "key" used to generate the forward Cauchy matrix is saved,
/// it can be used to calculate the inverse more efficiently than
/// doing full Gaussian elimination.
///
/// See:
/// * <https://en.wikipedia.org/wiki/Cauchy_matrix>
/// * <https://proofwiki.org/wiki/Inverse_of_Cauchy_Matrix>
///
/// Note that the inverse is a k\*k matrix, so 2\*k distinct values
/// must be passed in:
///
/// * the fixed `y1 .. yk` values
/// * a selection of k x values corresponding to the
///   k rows being combined

pub fn cauchy_inverse_matrix<G>
    (field : &G, key : &Vec<G::E>, k: usize) -> Vec<G::E>
where G : GaloisField
{
    let one  = G::E::one();

    let mut v = Vec::<G::E>::new();

    let vlen = key.len();
    if vlen != 2 * k {
	println!("Must supply k y values and k x values");
	return v
    }

    // slice as before
    let y : &[G::E] = &key[..k];
    let x : &[G::E] = &key[k..];

    // the reference version works, but the optimised version
    // doesn't. Only enabling reference version for now.
    if k != 0 {		      // was k < 3
	// this is the basic reference algorithm
	let n = k;		// alias to use i, j, k for loops
	for i in 0..n {		// row
	    for j in 0..n {	// column
		let mut top = one;
		for k in 0..n {
		    top = field.mul(top, x[j] ^ y[k]);
		    top = field.mul(top, x[k] ^ y[i]);
		}
		let mut bot = x[j] ^ y[i];
		for k in 0..n {
		    if k == j { continue }
		    bot = field.mul(bot, x[j] ^ x[k]);
		}
		for k in 0..n {
		    if k == i { continue }
		    bot = field.mul(bot, y[i] ^ y[k]);
		}
		top = field.mul(top, field.inv(bot));
		v.push(top);	// row-wise
	    }
	}

    } else {
	// TODO: find bug in this code
	//
	// optimised version of the above that notes that we
	// calculate the following in the inner loop:
	//
	// * product of row except for ... (n-1 multiplications)
        // * product of col except for ... (n-1 multiplications)
	//
	// These can be calculated outside the main loop and reused.
	//
	//
	let n = k;
	let mut imemo = Vec::<G::E>::with_capacity(k);
	for i in 0..n {
	    let mut bot = one;
	    let yi = y[i];
	    for k in 0..n {
		if k == i { continue }
		bot = field.mul(bot, yi ^ y[k]);
	    }
	    imemo.push(bot)
	}
	let mut jmemo = Vec::<G::E>::with_capacity(k);
	for j in 0..n {
	    let mut bot = one;
	    let xj = x[j];
	    for k in 0..n {
		if k == j { continue }
		bot = field.mul(bot, xj ^ x[k]);
	    }
	    jmemo.push(bot)
	}
	for i in 0..n {
	    for j in 0..n {
		let mut top = field.mul(x[j] ^ y[0], x[0] ^ y[i]);
		for k in 0..n {
		    top = field.mul(top, x[j] ^ y[k]);
		    top = field.mul(top, x[k] ^ y[i]);
		}
		let mut bot = x[j] ^ y[i];
		// inner loops eliminated:
		bot = field.mul(bot, imemo[i]);
		bot = field.mul(bot, jmemo[j]);
		top = field.mul(top, field.inv(bot));
		v.push(top)
	    }
	}
    }
    v
}


// I guess that I might have to implement a matrix here for now. The
// easiest way to test that the two Cauchy fns are correct is to
// calculate the forward and inverse matrices from the same key,
// multiply them together and check if we have an identity matrix.
//
// I'll need a matrix inversion routine if I want to check the
// correctness of the Vandermonde matrix code, though. I wonder,
// though, are there any short-cuts to calculating the inverse of
// that? 

// Can't derive Copy due to Vec not implementing it. Will need to
// implement a copy constructor so.
// #[derive(Debug)]
// pub struct Matrix<T, P>
// where T : NumericOps, P : NumericOps {
//     rows : usize,
//     cols : usize,
//     data : Vec<T>,
//     _phantom : P,
//     rowwise : bool,
// }

// Vector stuff done in guff crate
/*
// vector product ... multiply each vector element -> new vector
fn vector_mul<G>(field : &G,
		   dst : &mut [G::E], a : &[G::E], b : &[G::E])
where G : GaloisField {
    //    let prod = T::one;
    let (mut a_iter, mut b_iter) = (a.iter(), b.iter());
    for d in dst.iter_mut() {
	*d = field.mul(*a_iter.next().unwrap(), *b_iter.next().unwrap())
    }
}

// dot product ... sum of items in vector product -> value
fn dot_product<G>(field : &G,
		    a : &[G::E], b : &[G::E]) -> G::E
where G : GaloisField {
    let mut sum = G::E::zero();
    for (a_item, b_item) in a.iter().zip(b) {
	sum = sum ^ field.mul(*a_item, *b_item);
    }
    sum
}
 */

// Matrix stuff done in guff-matrix

// I think that I'll make rowwise and colwise matrices different
// types. Different type constraints, anyway. This makes sense for
// matrices that are passed into a multiplication routine, but I'm not
// sure about return values...
//
// There are two patterns I'm interested in.
//
// Split pattern:
//
//      k          window              window
//    +----+    +----- … -----+    +----- … -----+
//    |->  |    ||            |    |->           | > contiguous
//  n |    |  k |v            |  n |->           |
//    |    |    +----- … -----+    |             |
//    +----+     v contiguous      +----- … -----+
//
//  transform   x    input       =       output
//
//    ROWWISE       COLWISE             ROWWISE
//
//
// Combine pattern:
//
//      k          window              window
//    +----+    +----- … -----+    +----- … -----+
//    |->  |    |->           |    ||            | v contiguous
//  k |    |  k |->           |  k |v            |
//    +----+    +----- … -----+    +----- … -----+
//               > contiguous
//
//  transform   x    input       =       output
//
//    ROWWISE        ROWWISE            COLWISE
//
//
// The arrows show the optimal organisation for I/O: not necessarily
// for actual matrix multiplication.
//
// viz.:
//
// When splitting, single input stream is read into a contiguous block
// of memory, one column at a time, while several output streams are
// also contiguous.
//
// When combining, each of the several (rowwise) input streams are
// contiguous, and the single (colwise) output stream is contiguous.
//
// So basically the split version has an unavoidable scatter pattern
// at the output, while the combine has an unavoidable gather pattern
// at the input. We could try out some buffering strategy that
// internally transposes sub-matrix blocks (storing them in SIMD
// registers and working on them there) and translating their
// reads/writes so that they're in the "correct" form in
// memory. That's for another day, though.
//
// The most important thing from the above is that (ignoring the
// organisation of the transform matrix, which is always ROWWISE) our
// multiply will always take input data in one layout and output a
// matrix of the opposite layout. That basically answers my question
// about whether it's a good idea to have ROWWISE/COLWISE variants of
// matrices. It does seem to be.
//

// sketch of the above idea...

/*
struct RowWiseAdaptedMatrix<T> {
    // copy of all fields that a regular Matrix has, except layout
}
struct ColWiseAdaptedMatrix<T> {
    // copy of all fields that a regular Matrix has, except layout
}

trait MatrixOps<T> {
    // default, non-specialised stuff
    fn rows();
    // all the accessors and regular stuff that a Matrix does above.

    // Then, gaps for specialised (composable) methods below:
    fn kernel();
}
impl MatrixOps<T,P> {
    // default implementations
    
}

impl<T> MatrixOps<T> for RowWiseAdaptedMatrix<T>   {

    fn kernel () {
	// specialised kernel goes here
    }
}

// Another way to do it is to just use type constraints on
// functions. That is probably simpler:

fn split<T,P>(field : &impl GenericField<T,P>,
	      transform<T> : R, input<T> : C, output<T> : R)
where T : NumericOps,
      P : NumericOps,
      R : MatrixOps + RowWise,
      C : MatrixOps + ColWise
{
    // write split-specific kernel here
}

// This would be supported by:
// * empty traits for RowWise and ColWise
// * two specialised structs that have the traits composed in
// * different constructors for giving us concrete instances
// * coordination between other matrix fns that always make
//   clear which subtypes of matrix they expect/provide
// * helper functions for swapping (transposing) or reinterpreting(?)
//   matrix subtype
 */

// comment out matrix code

/*
// Put Accessors into a separate trait. I'm trying to see if I can get
// a default implementation that has the same members. Mmm... probably
// not. I suppose that a macro is the only solution to avoiding all
// the boilerplate.
// Needs to be generic on T due to dealing with Vec<T>
pub trait Accessors<T : NumericOps> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn rowcol(&self) -> (usize, usize) {
	(self.rows(), self.cols())
    }

    // internally, assume all implementations work with Vec<T>
    fn vec(&self)                   -> &Vec<T>;
    fn vec_as_mut(&mut self)        -> &mut Vec<T>;
    fn vec_as_mut_slice(&mut self)  -> &[T];
}


// Things that we can do with a matrix without knowing its layout
pub trait LayoutAgnostic<T : NumericOps> : Accessors<T>
{
    fn zero(&mut self) {
	let slice : &mut[T] = &mut self.vec_as_mut()[..];
	for elem in slice.iter_mut() { *elem = T::zero() }
    }
    fn one(&mut self) {		// sets matrix as identity
	let (rows, cols) = self.rowcol();
	if rows != cols {
	    panic!("Must have rows==cols to set up as identity matrix")
	}
	let slice : &mut[T] = &mut self.vec_as_mut()[..];
	let diag  = rows + 1;	// count mod this
	let mut index = 0;
	for elem in slice.iter_mut() {
	    if index == 0 {
		*elem = T::one()
	    } else {
		*elem = T::zero()
	    }
	    index = (index + 1) % diag
	}
    }

    // will have external transpose() for non-square matrices which
    // will create a new matrix with a new layout. This one flips
    // values along the diagonal without changing the layout
    fn transpose_square(&mut self) {
	let rows = self.rows();
	if rows != self.cols() {
	    panic!("Matrix is not square")
	}
	let v = self.vec_as_mut();
	// swap values at i,j with j,i for all i != j
	// we don't need to know anything about layout
	for i in 1..rows {
	    for j in 0..i {
		// safe way of swapping values without temp variable?
		// should be...
		// (v[i * rows + j], v[j * rows + i])
		// = (v[j * rows + i], v[i * rows + j])
		let t = v[i * rows + j];
		v[i * rows + j] = v[j * rows + i];
		v[j * rows + i] = t;
	    }
	}
    }
}

// Things that require knowing about the matrix's layout
pub trait LayoutSpecific<T : NumericOps> : Accessors<T> {

    // These essentially fix the layout
    fn is_rowwise(&self) -> bool;
    fn is_colwise(&self) -> bool { ! &self.is_rowwise() }
    fn _rowcol_to_index(&self, row : usize, col : usize) -> usize;

    // cursor/index pointer into vec (derived from above; no overflow
    // checks since Vec will catch it). The use case for these is to
    // make one initial call to whichever/both you want to use,
    // setting the index to 0. Then, in your loops, simply add the
    // correct offset. Calling move_* inside your loop isn't too bad
    // either assuming LLVM can notice that self.rows()/.cols() is a
    // loop invariant. It might even do the same for .is_rowwise().
    fn move_right(&self, index : usize) -> usize {
	if self.is_rowwise() { index + 1 } else { index + self.rows() }
    }
    fn move_down(&self, index : usize) -> usize {
	if self.is_rowwise() { index + self.cols() } else { index + 1 }
    }

    // higher-level users of the first three
    fn getval(&self, row : usize, col : usize) -> T
    {
	self.vec()[self._rowcol_to_index(row, col)]
    }
    fn setval(&mut self, row : usize, col : usize, val : T)
    {
	// need to use two statements below because you can't use
	// self both mutably (updating the vector) and immutably
	let index = self._rowcol_to_index(row, col);
	self.vec_as_mut()[index] = val;
    }
}

// What do they call fns in rust that are called with
// module::something()?  Whatever they're called, it probably makes
// more sense to call things like matrix multiply that way

/* 
// All variants will use the same structure format, but we give them
// different names to distinguish them
struct RowwiseMatrix<T : NumericOps> {
    rows : usize,
    cols : usize,
    v  : Vec<T>,
}
impl<T : NumericOps> Accessors<T> for RowwiseMatrix<T> {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn vec_as_mut_slice(&self) -> &[T]     { &self.v[..] }
    fn vec_as_mut(&mut self) ->       &mut Vec<T>  { &mut (self.v) }
}
struct ColwiseMatrix<T : NumericOps> {
    rows : usize,
    cols : usize,
    v :  Vec<T>,
}
impl<T : NumericOps> Accessors<T> for ColwiseMatrix<T> {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn vec_as_mut_slice(&self) -> & [T] { &self.v[..] }
    fn vec_as_mut(&mut self) -> &mut Vec<T>     { &mut (self.v) }
}

*/
pub struct CheckedMatrix<T : NumericOps> {
    rows : usize,
    cols : usize,
    v  : Vec<T>,
    is_rowwise : bool,		// checked at run time
}
impl<T : NumericOps> Accessors<T> for CheckedMatrix<T> {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn vec(&self)                  -> &Vec<T>    { &self.v }
    fn vec_as_mut(&mut self)       -> &mut Vec<T>    { &mut (self.v) }
    fn vec_as_mut_slice(&mut self) -> &[T] { &self.v[..] }
}
impl<T : NumericOps> LayoutAgnostic<T> for CheckedMatrix<T> { }
impl<T : NumericOps> LayoutSpecific<T> for CheckedMatrix<T> {
    fn is_rowwise(&self) -> bool {   self.is_rowwise }
    fn _rowcol_to_index(&self, row : usize, col : usize) -> usize {
	// won't check that row/col are within allowed range
	if self.is_rowwise {
	    row * self.cols + col
	} else {
	    col * self.rows + row
	}
    }
}

// what should our return type be?
//
// * just the struct type?
// * an impl line?
// x combine both? (can't have -> Struct : Foo Bar)

fn construct_checked_matrix<T : NumericOps>
    (rows : usize, cols : usize, rowwise : bool)
     -> CheckedMatrix<T>
//    where T: NumericOps, M : Accessors<T> + LayoutAgnostic<T> + LayoutSpecific<T>
{
    let v  = vec![T::zero(); rows * cols];
    
    CheckedMatrix::<T> {	// gains a concrete type
	rows : rows, cols : cols,
	v : v, is_rowwise : rowwise
    }
}

*/

// Following on from the names above, I think I'll rename the two
// traits that each struct should implement according to whether it's
// layout-neutral or layout-specific/-sensitive.

// Is this a good approach?
//
// I think it's OK. In "normal" OO, we'd have a base "Matrix" class
// from which we'd extend into row-wise and column-wise forms. Then
// the constructors would return types that are still classed as
// "Matrix" objects, no matter what the subclass name.
//
// We can't do that in Rust because there's no object inheritance.
//
// What we can do is return an object that satisfies trait interfaces.
//
// A problem arises when we want to store different struct variants in
// a single vector or variable. Even if all the variants are the same
// size, a return type of "implements Foo" does not guarantee that,
// and the type system can't go and check.
//
// Actually, though, that's also a problem in standard OO. The
// solution there is that we allocate objects on the heap and store
// pointers to them.
//
// I think that the way I'm handling it is the proper Rust way of
// doing things. For example, if the distinction between the different
// object variants matters, we can call the appropriate constructor
// (or manually construct the object) and store it with the
// appropriate struct type. As an example, a "transform" object that
// does encoding or decoding using IDA or Reed-Solomon coding will
// probably want to use specific row-wise or column-wise variants for
// the input, transform and output matrices.
//
// If the application doesn't care about the internal details, it can
// cast the returned object as something that "implements matrix
// operations on <some numeric type>". It loses the ability to
// ascertain the exact type of the underlying struct from that point
// on, though.
//
// There is a bit of extra boilerplate for all the three matrix types
// above. The accessors have to be attached to the structs (because
// traits deal with methods, not attributes)


#[cfg(test)]


mod tests {
    use super::*;
    // use num_traits::identities::{One,Zero};

    // External crate only used in development/testing
    use guff_matrix::*;
    use guff_matrix::x86::*;

    #[test]
    fn vandermonde_works() {
	let f = new_gf8(0x11b, 0x1b);
        let v = vandermonde_matrix(&f, 5, 5);
	for i in 5..10 {
	    assert_eq!(1, v[i], "expect row 1 all 1s");
	}
	for i in 0..5 {
	    // check column 0, including 0**0 = 1
	    assert_eq!(1, v[i * 5], "expect col 0 all 1s");
	    // check column 1
	    let index = i * 5 + 1;
	    assert_eq!(i, v[index].into(), "expect col 1 ascending 0..n");
	}
    }

    // The only sure-fire way to test the functions is to do a matrix
    // multiplication, then do the inverse and check whether the
    // result is the same as the original...
    //
    // I should have most of what I need in guff-matrix crate

    const SAMPLE_DATA : [u8; 20] = [
	1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ];

    // Direct copy from guff_matrix::simulator
    pub fn interleave_streams(dest : &mut [u8], slices : &Vec<&[u8]>) {

	let cols = dest.len() / slices.len();
	let mut dest = dest.iter_mut();
	let mut slice_iters : Vec::<_> = Vec::with_capacity(slices.len());
	for s in slices {
	    let iter = s.iter();
	    slice_iters.push(iter);
	}

	for _ in 0 .. cols {
	    for slice in &mut slice_iters {
		*dest.next().unwrap() = *slice.next().unwrap();
	    }
	}
    }

    #[test]
    fn test_cauchy_transform() {

	// 4x4 xform matrix is the smallest allowed right now
	let key = vec![ 1, 2, 3, 4, 5, 6, 7, 8 ];	
	let field = new_gf8(0x11b, 0x1b);
	let cauchy_data = cauchy_matrix(&field, &key, 4, 4);

	// guff-matrix needs either/both/all:
	// * architecture-neutral matrix type (NoSimd option)
	// * automatic selection of arch-specific type (with fallback)
	// * new_matrix() type constructors?
	//
	// For now, though, the only concrete matrix types that are
	// implemented are for supporting x86 simd operation, so use
	// those (clunky) names...

	let mut xform = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,4,true);
	xform.fill(&cauchy_data);

	// must choose cols appropriately (gcd requirement)
	let mut input = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,5,false);
	input.fill(&SAMPLE_DATA);

	let mut output = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,5,true);

	// use non-SIMD multiply
	
	reference_matrix_multiply(&mut xform, &mut input,
				  &mut output, &field);

	// We will also need a transposition (interleaving) step to
	// convert the rowwise output matrix from above into colwise
	// format for the inverse transform

	// Right now, only implementation of interleaver is in the
	// simulation module... copying it in here

	// we need to do some up-front work to use that:
	let array = output.as_slice();
	let slices : Vec<&[u8]> = array.chunks(5).collect();
	let mut dest = [0u8; 20];
	
	interleave_streams(&mut dest, &slices);

	// Do the inverse transform (same key)

	let cauchy_data = cauchy_inverse_matrix(&field, &key, 4);
	let mut xform = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,4,true);
	xform.fill(&cauchy_data);

	let mut input = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,5,false);
	input.fill(&dest);

	let mut output = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,5,false);

	// use non-SIMD multiply
	reference_matrix_multiply(&mut xform, &mut input,
				  &mut output, &field);

	assert_eq!(output.as_slice(), &SAMPLE_DATA);

	unsafe {
	    // use SIMD multiply
	    simd_warm_multiply(&mut xform, &mut input,
			       &mut output);
	}

	assert_eq!(output.as_slice(), &SAMPLE_DATA);

	
    }

    // Another test I could do would be to multiply the Cauchy matrix
    // by its inverse and check that the result is the identity
    // matrix. However, as it currently stands, guff-matrix doesn't
    // allow for general-purpose matrix multiply yet, since it's
    // optimised for xform/input matrix pairs that satisfy the gcd
    // property.
    //
    // Ah, I think I can ... no gcd checks in matrix_multiply
    #[test]
    fn test_inv_inv_cauchy() {
	// 4x4 xform matrix is the smallest allowed right now
	let key = vec![ 1, 2, 3, 4, 5, 6, 7, 8 ];	
	let field = new_gf8(0x11b, 0x1b);

	let forward_data = cauchy_matrix(&field, &key, 4, 4);
	let mut xform = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,4,true);
	xform.fill(&forward_data);

	// data returned from cauchy_inverse_matrix needs to be
	// interleaved if it's in the 'input' position

	let inverse_data = cauchy_inverse_matrix(&field, &key, 4);

	let array = inverse_data;
	let slices : Vec<&[u8]> = array.chunks(4).collect();
	let mut dest = [0u8; 16];

	interleave_streams(&mut dest, &slices);
	
	let mut input = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,4,false);
	input.fill(&dest);

	let mut output = X86SimpleMatrix::<x86::X86u8x16Long0x11b>
	    ::new(4,4,true);

	// use non-SIMD multiply
	reference_matrix_multiply(&mut xform, &mut input,
				  &mut output, &field);

	assert_eq!(output.as_slice(),
		   [ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 ]
	)

    }

    
    // Can't test Vandermonde yet because I haven't implemented matrix
    // inversion here or in guff-matrix
    
    /*
    // Might as well start writing some test cases
    #[test]
    fn test_making_vectors() {
	// can we convert u8 to T?
	let a = [0u8,1,2,3,4];
	// do I have to clone() to prevent getting refs?
	let v1 : Vec<u8> = a.iter().cloned().collect();
	let v2 : Vec<u8> = a.iter().cloned().collect();
	let mut v3 : Vec<u8> = a.iter().cloned().collect();
	let f = new_gf8(0x11b,0x1b);

	// should change v3
	assert_eq!(v1, v3);
	vector_mul(&f, &mut v3, &v1, &v2);
	assert_ne!(v1, v3);
    }
     */

    /*

    #[test]
    fn test_dot_products() {
	// I'll only use mul by zero or one so that I can mentally
	// calculate the result without needing to know results of
	// more complex field multiplications
	let a = [0u8, 1, 1, 9, 4, 8, 4];
	let b = [0u8, 0, 2, 0, 1, 1, 1];
	// do I have to clone() to prevent getting refs?
	let v1 : Vec<u8> = a.iter().cloned().collect();
	let v2 : Vec<u8> = b.iter().cloned().collect();
	let f = new_gf8(0x11b,0x1b);

	// should change v3
	assert_ne!(v1, v2);
	let sum = dot_product(&f, &v1, &v2);
	assert_eq!(sum, 0 ^ 0 ^ 2 ^ 0 ^ 4 ^ 8 ^ 4);
    }
    #[test]
    fn test_construct_checked_matrix() {
	let mat = construct_checked_matrix::<u8>(3, 4, false);
	assert_eq!(3, mat.rows());
	assert_eq!(4, mat.cols());
	assert_eq!(false, mat.is_rowwise());
	assert_eq!(true , mat.is_colwise());
    }

    // fn sig that requires a certain trait be implemented
    fn looking_for_accessors<T>(mat : &impl Accessors<T>)
    where T : NumericOps {
	assert!(mat.rows() > 0)
    }

    // we get a struct, so pass it to looking_for_accessors() above 
    #[test]
    fn test_impl_accessors_satisfied() {
	let mat = construct_checked_matrix::<u8>(3, 4, false);
	// no compile error, and assert above passes
	looking_for_accessors(&mat);
    }

    // Various simple things to test ...
    //
    // * initial matrix is zeroed
    // * identity works
    // * we can compare output of vec accessor with a list

    #[test]
    fn basic_vec_comparison() {
	let mut mat = construct_checked_matrix::<u8>(3, 3, true);
	// it seems we have to compare as slice ... no big deal
	assert_eq!([0u8; 9], &mat.vec()[..]);

	mat.one();
	assert_eq!([1u8, 0, 0, 0, 1, 0, 0, 0, 1], &mat.vec()[..]);

	mat.zero();
	assert_eq!([0u8; 9], &mat.vec()[..]);

	// What if we compare shared ref with mutable slice?
	// surprisingly, this works with no compiler complaint.
	assert_eq!([0u8; 9], mat.vec_as_mut_slice());
    }

    #[test]
    fn mutate_single_vec_elements() {
	let mut mat = construct_checked_matrix::<u8>(3, 3, true);
	{			// drop v after use: see below
	    let mut v = mat.vec_as_mut();
	    v[0] = 1; v[4] = 1; v[8] = 1;
	}
	let mut identity = construct_checked_matrix::<u8>(3, 3, true);
	identity.one();		// maybe make this a fluent interface?

	assert_eq!(mat.vec(), identity.vec());

	// without the braces above, the assignment below would cause
	// the compiler to complain about mixing mutable/immutable
	// borrows. So we drop the original v and then recreate it
	// here after the immutable borrow in mat.vec() above.
	let mut v = mat.vec_as_mut();
	v[0] = 1;
    }

    // basic test of _rowcol_to_index (suffices for all matrix sizes)
    #[test]
    fn test_rowwise_checked() {
	// 2x2 matrix gives a truth table of sorts
	let rowwise = construct_checked_matrix::<u8>(2, 2, true);
	assert_eq!(0, rowwise._rowcol_to_index(0,0));
	assert_eq!(1, rowwise._rowcol_to_index(0,1));
	assert_eq!(2, rowwise._rowcol_to_index(1,0));
	assert_eq!(3, rowwise._rowcol_to_index(1,1));

    	let colwise = construct_checked_matrix::<u8>(2, 2, false);
	assert_eq!(0, colwise._rowcol_to_index(0,0));
	assert_eq!(1, colwise._rowcol_to_index(1,0)); // these two
	assert_eq!(2, colwise._rowcol_to_index(0,1)); // swapped
	assert_eq!(3, colwise._rowcol_to_index(1,1));
    }

    // "cursors" are another way to traverse rows/columns. If these
    // work, we don't need to check any larger matrix sizes.
    #[test]
    fn test_cursor_checked() {
	// 2x2 matrix gives a truth table of sorts
	let rowwise = construct_checked_matrix::<u8>(2, 2, true);
	assert_eq!(1, rowwise.move_right(0));
	assert_eq!(2, rowwise.move_down(0));

    	let colwise = construct_checked_matrix::<u8>(2, 2, false);
	assert_eq!(2, colwise.move_right(0));
	assert_eq!(1, colwise.move_down(0));
    }

    // Other stuff relating to orientation also needs checking. We
    // can't assume anything about how any other layout-specific
    // functions will reason about it, and which of the above
    // functions they use.
    //
    // I might refactor the above tests so that it's easier to reuse
    // them when I implement the other two (non-checked) layouts
    //
    // something like "assert_rowwise_cursor(mat)"

    // I can nearly test cauchy functions ... just need a multiply routine
    #[test]
    fn test_cauchy_inverse_identity() {
	todo!()
    }
*/
}
