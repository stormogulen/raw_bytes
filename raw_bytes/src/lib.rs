#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
pub use std::{vec::Vec, string::String};

#[cfg(not(feature = "std"))]
pub use alloc::{vec::Vec, string::String};

pub mod container;
pub mod error;
pub mod storage;

pub use container::Container;
pub use error::ContainerError;
pub use storage::Storage;
