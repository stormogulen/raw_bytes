
//! Conversion traits and utilities

use bytemuck::{Pod, cast_slice, cast_slice_mut};

/// Convert types to byte slices
pub trait ToBytes {
    fn to_bytes(&self) -> &[u8];
}

impl<T: Pod> ToBytes for [T] {
    fn to_bytes(&self) -> &[u8] {
        cast_slice(self)
    }
}

impl<T: Pod> ToBytes for Vec<T> {
    fn to_bytes(&self) -> &[u8] {
        cast_slice(self.as_slice())
    }
}

/// Convert byte slices to typed slices
pub trait FromBytes<T> {
    fn from_bytes(bytes: &[u8]) -> &[T];
    fn from_bytes_mut(bytes: &mut [u8]) -> &mut [T];
}

impl<T: Pod> FromBytes<T> for [T] {
    fn from_bytes(bytes: &[u8]) -> &[T] {
        cast_slice(bytes)
    }

    fn from_bytes_mut(bytes: &mut [u8]) -> &mut [T] {
        cast_slice_mut(bytes)
    }
}

/// Unified conversion trait for packed data
pub trait PackedConvert: Sized {
    fn pack_into(&self, buffer: &mut Vec<u8>);
    fn unpack_from(buffer: &[u8]) -> Option<Self>;
}

// Implement for Pod types automatically
impl<T: Pod + Copy> PackedConvert for T {
    fn pack_into(&self, buffer: &mut Vec<u8>) {
        let bytes = bytemuck::bytes_of(self);
        buffer.extend_from_slice(bytes);
    }

    fn unpack_from(buffer: &[u8]) -> Option<Self> {
        if buffer.len() >= std::mem::size_of::<T>() {
            Some(*bytemuck::from_bytes(&buffer[..std::mem::size_of::<T>()]))
        } else {
            None
        }
    }
}
