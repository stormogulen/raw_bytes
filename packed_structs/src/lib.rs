#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String};

use bytemuck::{Pod, Zeroable};

pub mod packed_bytes {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PackedBytes<const N: usize> {
        bytes: [u8; N],
    }

    unsafe impl<const N: usize> Zeroable for PackedBytes<N> {}
    unsafe impl<const N: usize> Pod for PackedBytes<N> {}

    impl<const N: usize> Default for PackedBytes<N> {
        fn default() -> Self { Self::new() }
    }

    impl<const N: usize> PackedBytes<N> {
        pub fn new() -> Self { Self { bytes: [0; N] } }
        pub fn from_bytes(bytes: [u8; N]) -> Self { Self { bytes } }
        pub fn as_bytes(&self) -> &[u8] { &self.bytes }
        pub fn as_bytes_mut(&mut self) -> &mut [u8] { &mut self.bytes }
        pub fn as_pod<T: Pod>(&self) -> &T {
            assert_eq!(std::mem::size_of::<T>(), N, "Type size mismatch");
            bytemuck::from_bytes(&self.bytes)
        }
        pub fn as_pod_mut<T: Pod>(&mut self) -> &mut T {
            assert_eq!(std::mem::size_of::<T>(), N, "Type size mismatch");
            bytemuck::from_bytes_mut(&mut self.bytes)
        }
        pub fn get<T: Pod + Copy>(&self) -> T { *self.as_pod::<T>() }
        pub fn set<T: Pod>(&mut self, value: T) {
            assert_eq!(std::mem::size_of::<T>(), N, "Type size mismatch");
            self.bytes.copy_from_slice(bytemuck::bytes_of(&value));
        }
    }

    pub fn cast_slice<T: Pod, const N: usize>(packed_slice: &[PackedBytes<N>]) -> &[T] {
        assert_eq!(std::mem::size_of::<T>(), N, "Type size mismatch");
        bytemuck::cast_slice(packed_slice)
    }

    pub fn cast_slice_mut<T: Pod, const N: usize>(packed_slice: &mut [PackedBytes<N>]) -> &mut [T] {
        assert_eq!(std::mem::size_of::<T>(), N, "Type size mismatch");
        bytemuck::cast_slice_mut(packed_slice)
    }
}

// Only include the container module if the feature is enabled
#[cfg(feature = "container")]
pub mod container;
