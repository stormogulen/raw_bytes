// src/container.rs - Updated to work with raw_bytes::Container

use raw_bytes::Container;
use crate::PackedBitsError;

#[cfg(not(feature = "std"))]
use alloc::vec;

const MAGIC: &[u8; 4] = b"PKBT";
const HEADER_SIZE: usize = 12;

#[derive(Debug)]
pub struct PackedBitsContainer<const N: usize> {
    storage: Container<u8>,
    len: usize,
}

type Result<T> = core::result::Result<T, PackedBitsError>;

impl<const N: usize> PackedBitsContainer<N> {
    pub fn new_in_memory() -> Self {
        assert!(N > 0 && N <= 32, "N must be 1..=32");
        let mut storage = Container::from_slice(&vec![0u8; HEADER_SIZE]);
        Self::write_header(&mut storage, 0).expect("failed to write header");
        Self { storage, len: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        assert!(N > 0 && N <= 32, "N must be 1..=32");
        let data_bytes = (capacity * N).div_ceil(8);
        let total_bytes = HEADER_SIZE + data_bytes;
        let mut storage = Container::from_slice(&vec![0u8; total_bytes]);
        Self::write_header(&mut storage, 0).expect("failed to write header");
        Self { storage, len: 0 }
    }

    pub fn from_storage(storage: Container<u8>) -> Result<Self> {
        assert!(N > 0 && N <= 32, "N must be 1..=32");
        if storage.len() < HEADER_SIZE {
            return Err(PackedBitsError::StorageTooSmall);
        }
        let slice = storage.as_slice();
        if &slice[0..4] != MAGIC {
            return Err(PackedBitsError::InvalidMagic);
        }
        let stored_n = u32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]);
        if stored_n as usize != N {
            return Err(PackedBitsError::InvalidN {
                expected: N,
                found: stored_n,
            });
        }
        let len = u32::from_le_bytes([slice[8], slice[9], slice[10], slice[11]]) as usize;
        Ok(Self { storage, len })
    }

    pub fn from_storage_raw(storage: Container<u8>) -> Self {
        let len_elements = (storage.len() * 8) / N;
        Self { storage, len: len_elements }
    }

    fn write_header(storage: &mut Container<u8>, len: usize) -> Result<()> {
        let slice = storage.as_slice();
        if slice.len() < HEADER_SIZE {
            return Err(PackedBitsError::StorageTooSmall);
        }
        // We need to modify the container, so we'll need to use indexing
        for i in 0..4 {
            storage[i] = MAGIC[i];
        }
        let n_bytes = (N as u32).to_le_bytes();
        for i in 0..4 {
            storage[4 + i] = n_bytes[i];
        }
        let len_bytes = (len as u32).to_le_bytes();
        for i in 0..4 {
            storage[8 + i] = len_bytes[i];
        }
        Ok(())
    }

    fn update_len_in_header(&mut self) -> Result<()> {
        let len_bytes = (self.len as u32).to_le_bytes();
        for i in 0..4 {
            self.storage[8 + i] = len_bytes[i];
        }
        Ok(())
    }

    pub fn storage(&self) -> &Container<u8> {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut Container<u8> {
        &mut self.storage
    }

    fn ensure_capacity(&mut self, total_bits: usize) -> Result<()> {
        let required_bytes = HEADER_SIZE + total_bits.div_ceil(8);
        let current_len = self.storage.len();
        
        if current_len < required_bytes {
            // Grow the container
            let additional = required_bytes - current_len;
            for _ in 0..additional {
                self.storage.push(0).map_err(|_| PackedBitsError::ResizeFailed)?;
            }
        }
        Ok(())
    }

    pub fn push(&mut self, value: u32) -> Result<()> {
        let max_val = if N == 32 { u32::MAX } else { (1u32 << N) - 1 };
        assert!(value <= max_val, "value must fit in {} bits", N);
        let bit_pos = self.len * N;
        self.ensure_capacity(bit_pos + N)?;
        let byte_pos = HEADER_SIZE + bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mut v = value as u64;
        v <<= bit_offset;
        
        let num_bytes = (N + bit_offset).div_ceil(8);
        debug_assert!(num_bytes <= 5);
        
        for i in 0..num_bytes {
            self.storage[byte_pos + i] |= ((v >> (i * 8)) & 0xFF) as u8;
        }
        
        self.len += 1;
        self.update_len_in_header()?;
        Ok(())
    }

    pub fn get(&self, index: usize) -> Option<u32> {
        if index >= self.len {
            return None;
        }
        let bit_pos = index * N;
        let byte_pos = HEADER_SIZE + bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mut val: u64 = 0;
        let slice = self.storage.as_slice();
        
        let num_bytes = (N + bit_offset).div_ceil(8);
        debug_assert!(num_bytes <= 5);
        
        for i in 0..num_bytes {
            if byte_pos + i < slice.len() {
                val |= (slice[byte_pos + i] as u64) << (i * 8);
            }
        }
        
        val >>= bit_offset;
        let mask = if N == 32 { u32::MAX as u64 } else { (1u64 << N) - 1 };
        Some((val & mask) as u32)
    }

    pub fn set(&mut self, index: usize, value: u32) -> Result<()> {
        assert!(index < self.len, "index out of bounds");
        let max_val = if N == 32 { u32::MAX } else { (1u32 << N) - 1 };
        assert!(value <= max_val, "value must fit in {} bits", N);
        
        let bit_pos = index * N;
        let byte_pos = HEADER_SIZE + bit_pos / 8;
        let bit_offset = bit_pos % 8;
        
        let mut v = value as u64;
        v <<= bit_offset;
        
        let mask: u64 = if N == 32 && bit_offset == 0 {
            u32::MAX as u64
        } else if N + bit_offset >= 64 {
            u64::MAX
        } else {
            ((1u64 << N) - 1) << bit_offset
        };
        
        let num_bytes = (N + bit_offset).div_ceil(8);
        debug_assert!(num_bytes <= 5);
        
        for i in 0..num_bytes {
            if byte_pos + i < self.storage.len() {
                let byte_mask = ((mask >> (i * 8)) & 0xFF) as u8;
                self.storage[byte_pos + i] &= !byte_mask;
                self.storage[byte_pos + i] |= ((v >> (i * 8)) & 0xFF) as u8;
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }


    pub fn clear(&mut self) -> Result<()> {
        self.len = 0;
        // Recreate storage with just the header
        self.storage = Container::from_slice(&vec![0u8; HEADER_SIZE]);
        self.update_len_in_header()?;
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        let data_bytes = self.storage.len().saturating_sub(HEADER_SIZE);
        (data_bytes * 8) / N
    }

    pub fn iter(&self) -> Iter<'_, N> {
        Iter { container: self, index: 0 }
    }
}

pub struct Iter<'a, const N: usize> {
    container: &'a PackedBitsContainer<N>,
    index: usize,
}

impl<'a, const N: usize> Iterator for Iter<'a, N> {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.container.len() {
            None
        } else {
            let val = self.container.get(self.index);
            self.index += 1;
            val
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.container.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, const N: usize> ExactSizeIterator for Iter<'a, N> {}

impl<'a, const N: usize> IntoIterator for &'a PackedBitsContainer<N> {
    type Item = u32;
    type IntoIter = Iter<'a, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_in_memory() {
        let mut pb = PackedBitsContainer::<12>::new_in_memory();
        pb.push(0xABC).unwrap();
        pb.push(0x123).unwrap();
        pb.push(0xFFF).unwrap();
        assert_eq!(pb.len(), 3);
        assert_eq!(pb.get(0), Some(0xABC));
        assert_eq!(pb.get(1), Some(0x123));
        assert_eq!(pb.get(2), Some(0xFFF));
        pb.set(1, 0x456).unwrap();
        assert_eq!(pb.get(1), Some(0x456));
        
        #[cfg(not(feature = "std"))]
        let collected: alloc::vec::Vec<_> = pb.iter().collect();
        #[cfg(feature = "std")]
        let collected: std::vec::Vec<_> = pb.iter().collect();
        
        #[cfg(not(feature = "std"))]
        assert_eq!(collected, alloc::vec![0xABC, 0x456, 0xFFF]);
        #[cfg(feature = "std")]
        assert_eq!(collected, vec![0xABC, 0x456, 0xFFF]);
    }

    #[test]
    fn test_header_persistence() {
        let mut pb = PackedBitsContainer::<7>::new_in_memory();
        pb.push(100).unwrap();
        pb.push(50).unwrap();
        let bytes = pb.storage().as_slice().to_vec();
        let storage = Container::from_slice(&bytes);
        let pb2 = PackedBitsContainer::<7>::from_storage(storage).unwrap();
        assert_eq!(pb2.len(), 2);
        assert_eq!(pb2.get(0), Some(100));
        assert_eq!(pb2.get(1), Some(50));
    }

    #[test]
    fn test_n32() {
        let mut pb = PackedBitsContainer::<32>::new_in_memory();
        pb.push(u32::MAX).unwrap();
        pb.push(12345).unwrap();
        assert_eq!(pb.get(0), Some(u32::MAX));
        assert_eq!(pb.get(1), Some(12345));
    }

    #[test]
    fn test_clear() {
        let mut pb = PackedBitsContainer::<5>::new_in_memory();
        pb.push(10).unwrap();
        pb.push(20).unwrap();
        assert_eq!(pb.len(), 2);
        pb.clear().unwrap();
        assert_eq!(pb.len(), 0);
        assert!(pb.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let mut pb = PackedBitsContainer::<8>::with_capacity(100);
        assert!(pb.capacity() >= 100);
        assert_eq!(pb.len(), 0);
        for i in 0..50 {
            pb.push(i as u32).unwrap();
        }
        assert_eq!(pb.len(), 50);
    }

    #[test]
    fn test_wrong_n() {
        let mut pb = PackedBitsContainer::<7>::new_in_memory();
        pb.push(100).unwrap();
        let bytes = pb.storage().as_slice().to_vec();
        let storage = Container::from_slice(&bytes);
        let result = PackedBitsContainer::<12>::from_storage(storage);
        assert!(matches!(result, Err(PackedBitsError::InvalidN { expected: 12, found: 7 })));
    }
}