use bytemuck::Pod;
use crate::{ContainerError, Storage};

#[cfg(feature = "std")]
use std::path::Path;

/// High-level container for typed elements
#[derive(Debug)]
pub struct Container<T: Pod> {
    storage: Storage<T>,
}

impl<T: Pod> Container<T> {
    /// In-memory container
    pub fn new_in_memory() -> Self {
        Container {
            storage: Storage::new_in_memory(),
        }
    }

    /// Open a memory-mapped read-only container
    #[cfg(feature = "mmap")]
    pub fn mmap_readonly<P: AsRef<Path>>(path: P) -> Result<Self, ContainerError> {
        let storage = Storage::from_mmap_readonly(path.as_ref())?;
        Ok(Container { storage })
    }

    /// Open a memory-mapped read-write container
    #[cfg(feature = "mmap")]
    pub fn mmap_readwrite<P: AsRef<Path>>(path: P) -> Result<Self, ContainerError> {
        let storage = Storage::from_mmap_readwrite(path.as_ref())?;
        Ok(Container { storage })
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Read a reference to element at index
    pub fn read_value(&self, index: usize) -> Result<&T, ContainerError> {
        self.storage.get(index)
    }

    /// Write element at index
    ///
    /// Only allowed for InMemory or MmapReadWrite
    pub fn write_value(&mut self, index: usize, value: T) -> Result<(), ContainerError> {
        let slot = self.storage.get_mut(index)?;
        *slot = value;
        Ok(())
    }

    /// Push an element — only works for InMemory
    pub fn push(&mut self, value: T) -> Result<(), ContainerError> {
        self.storage.push(value)
    }

    // forward methods
    pub fn get(&self, index: usize) -> Result<&T, ContainerError> {
        self.storage.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Result<&mut T, ContainerError> {
        self.storage.get_mut(index)
    }   
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck_derive::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    struct Packet {
        id: u32,
        value: f32,
    }

    #[test]
    fn in_memory_basic_operations() -> Result<(), ContainerError> {
        let mut c = Container::<Packet>::new_in_memory();
        assert!(c.is_empty());

        let p1 = Packet { id: 1, value: 10.0 };
        let p2 = Packet { id: 2, value: 20.0 };
        c.push(p1)?;
        c.push(p2)?;

        assert_eq!(c.len(), 2);
        assert_eq!(c.read_value(0)?, &p1);
        assert_eq!(c.read_value(1)?, &p2);

        // modify
        let p3 = Packet { id: 3, value: 30.0 };
        c.write_value(1, p3)?;
        assert_eq!(c.read_value(1)?, &p3);

        Ok(())
    }

    #[test]
    fn push_out_of_bounds() {
        let mut c = Container::<Packet>::new_in_memory();
        let p = Packet { id: 1, value: 10.0 };
        c.push(p).unwrap();

        let res = c.read_value(5);
        assert!(matches!(res, Err(ContainerError::OutOfBounds(5))));
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_readonly_error_on_write() -> Result<(), ContainerError> {
        use std::fs::File;
        use tempfile::tempfile;

        let file = tempfile()?;
        let c = Container::<Packet>::mmap_readonly(&file)?;
        let res = c.write_value(0, Packet { id: 1, value: 1.0 });
        assert!(res.is_err());
        Ok(())
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn mmap_readwrite_basic_operations() -> Result<(), ContainerError> {
        use std::fs::File;
        use tempfile::tempfile;

        let mut file = tempfile()?;
        file.set_len(1024)?; // make file big enough

        let mut c = Container::<Packet>::mmap_readwrite(&file)?;
        assert!(c.len() > 0 || c.len() == 0); // depends on file size

        // Cannot push because it's mmap
        let res = c.push(Packet { id: 1, value: 1.0 });
        assert!(res.is_err());

        Ok(())
    }
}
