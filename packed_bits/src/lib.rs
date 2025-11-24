//! # packed_bits
//!
//! A `no_std` compatible bit-packing library.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "container", not(feature = "std")))]
extern crate alloc;

pub mod error;
pub use error::PackedBitsError;

#[cfg(feature = "container")]
pub mod container;

#[cfg(feature = "container")]
pub mod flags;

#[cfg(feature = "container")]
pub use container::PackedBitsContainer;

#[cfg(feature = "container")]
pub use flags::FlagsContainer;