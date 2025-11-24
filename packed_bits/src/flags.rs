//! Flags container for bit flags

use crate::{PackedBitsContainer, PackedBitsError};

type Result<T> = core::result::Result<T, PackedBitsError>;

#[derive(Debug)]
pub struct FlagsContainer<const N: usize> {
    bits: PackedBitsContainer<N>,
}

impl<const N: usize> FlagsContainer<N> {
    pub fn new_in_memory() -> Self {
        Self { bits: PackedBitsContainer::<N>::new_in_memory() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { bits: PackedBitsContainer::<N>::with_capacity(capacity) }
    }

    pub fn push(&mut self, flags: u32) -> Result<()> {
        self.bits.push(flags)
    }

    pub fn contains(&self, index: usize, mask: u32) -> bool {
        self.bits.get(index).is_some_and(|val| (val & mask) != 0)
    }

    pub fn set_mask(&mut self, index: usize, mask: u32) -> Result<()> {
        if let Some(val) = self.bits.get(index) {
            let new_val = val | mask;
            self.bits.set(index, new_val)
        } else {
            Err(PackedBitsError::StorageTooSmall)
        }
    }

    pub fn clear_mask(&mut self, index: usize, mask: u32) -> Result<()> {
        if let Some(val) = self.bits.get(index) {
            let new_val = val & !mask;
            self.bits.set(index, new_val)
        } else {
            Err(PackedBitsError::StorageTooSmall)
        }
    }

    pub fn toggle_mask(&mut self, index: usize, mask: u32) -> Result<()> {
        if let Some(val) = self.bits.get(index) {
            let new_val = val ^ mask;
            self.bits.set(index, new_val)
        } else {
            Err(PackedBitsError::StorageTooSmall)
        }
    }

    pub fn get(&self, index: usize) -> Option<u32> {
        self.bits.get(index)
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn clear(&mut self) -> Result<()> {
        self.bits.clear()
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits.iter()
    }

    pub fn packed_bits(&self) -> &PackedBitsContainer<N> {
        &self.bits
    }

    pub fn iter_flags(&self, index: usize) -> Option<FlagsIter> {
        self.get(index).map(FlagsIter::new)
    }
}

pub struct FlagsIter {
    bits: u32,
    next_mask: u32,
}

impl FlagsIter {
    pub fn new(bits: u32) -> Self {
        Self { bits, next_mask: 1 }
    }
}

impl Iterator for FlagsIter {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        while self.next_mask != 0 {
            let mask = self.next_mask;
            self.next_mask <<= 1;
            if (self.bits & mask) != 0 {
                return Some(mask);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAG0: u32 = 1 << 0;
    const FLAG1: u32 = 1 << 1;
    const FLAG2: u32 = 1 << 2;

    #[test]
    fn basic_flags_ops() -> Result<()> {
        let mut fc = FlagsContainer::<3>::new_in_memory();
        fc.push(FLAG0 | FLAG2)?;
        fc.push(FLAG1)?;
        assert!(fc.contains(0, FLAG0));
        assert!(!fc.contains(0, FLAG1));
        fc.set_mask(1, FLAG2)?;
        assert_eq!(fc.get(1).unwrap(), FLAG1 | FLAG2);
        Ok(())
    }

    #[test]
    fn iter_flags_works() -> Result<()> {
        let mut fc = FlagsContainer::<3>::new_in_memory();
        fc.push(FLAG0 | FLAG2)?;
        fc.push(FLAG1)?;
        
        #[cfg(not(feature = "std"))]
        let mut all_flags = alloc::vec![];
        #[cfg(feature = "std")]
        let mut all_flags = vec![];
        
        for i in 0..fc.len() {
            if let Some(it) = fc.iter_flags(i) {
                #[cfg(not(feature = "std"))]
                all_flags.push(it.collect::<alloc::vec::Vec<u32>>());
                #[cfg(feature = "std")]
                all_flags.push(it.collect::<Vec<u32>>());
            }
        }
        
        #[cfg(not(feature = "std"))]
        assert_eq!(all_flags, alloc::vec![alloc::vec![FLAG0, FLAG2], alloc::vec![FLAG1]]);
        #[cfg(feature = "std")]
        assert_eq!(all_flags, vec![vec![FLAG0, FLAG2], vec![FLAG1]]);
        
        Ok(())
    }
}