A simple container for storing and accessing Plain Old Data (POD) types in memory or memory-mapped files.
Features

- Type-safe storage for any T: bytemuck::Pod type
- Multiple backends:

    -- In-memory Vec<T> for dynamic data
    -- Memory-mapped files for large datasets and persistence


- Zero-copy access with efficient slice operations
- Flexible - read-only or read-write mmap support
- no_std compatible (with alloc, std optional)

Basic In-Memory Storage

use raw_bytes::Container;
use bytemuck_derive::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Packet {
    id: u32,
    value: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create container
    let mut container = Container::<Packet>::new();
    
    // Add elements
    container.push(Packet { id: 1, value: 10.0 })?;
    container.push(Packet { id: 2, value: 20.0 })?;
    
    // Access elements
    let packet = container.get(0)?;
    println!("Packet: {:?}", packet);
    
    // Modify elements
    container.write(1, Packet { id: 2, value: 30.0 })?;
    
    // Iterate efficiently
    for packet in container.iter() {
        println!("id: {}, value: {}", packet.id, packet.value);
    }
    
    Ok(())
}

Memory-Mapped Files
Work with large datasets that don't fit in memory:

use raw_bytes::Container;
use std::io::Write;

#[cfg(feature = "mmap")]
fn process_large_dataset() -> Result<(), Box<dyn std::error::Error>> {
    // Create a data file
    let mut file = std::fs::File::create("data.bin")?;
    let data = vec![Packet { id: 1, value: 10.0 }; 1_000_000];
    file.write_all(bytemuck::cast_slice(&data))?;
    drop(file);
    
    // Memory-map for fast access (read-only)
    let container = Container::<Packet>::mmap_readonly("data.bin")?;
    println!("Loaded {} packets", container.len());
    
    // Zero-copy iteration over millions of elements
    let sum: f32 = container.iter().map(|p| p.value).sum();
    println!("Sum: {}", sum);
    
    Ok(())
}

#[cfg(feature = "mmap")]
fn update_persistent_data() -> Result<(), Box<dyn std::error::Error>> {
    // Open for read-write (changes persist to disk)
    let mut container = Container::<Packet>::mmap_readwrite("data.bin")?;
    
    // Modify data
    container.write(0, Packet { id: 99, value: 999.0 })?;
    
    // Changes are automatically synced to file
    Ok(())
}

Bulk Operations

// Pre-allocate for better performance
let mut container = Container::<Packet>::with_capacity(10_000);

// Create from existing data
let data = vec![Packet { id: 1, value: 1.0 }; 100];
let container = Container::from_slice(&data);

// Bulk operations using slices
let slice = container.as_slice();
let total: f32 = slice.iter().map(|p| p.value).sum();


Feature Flags
FeatureDefaultDescriptionstd✓Enables standard library support (required for files)mmap✗Enables memory-mapped file support (requires std)

# Minimal (no_std with alloc)
raw_bytes = { version = "0.1", default-features = false }

# With mmap support
raw_bytes = { version = "0.1", features = ["mmap"] }