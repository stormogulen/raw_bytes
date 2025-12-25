//use raw_bytes::Container;
//use bytemuck::Pod;
use bytemuck_derive::Pod;
use bytemuck_derive::Zeroable;
#[cfg(feature = "std")]
use std::{fs::File, io::Write, path::Path};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Packet {
    id: u32,
    value: f32,
}

#[cfg(feature = "std")]
fn ensure_file(path: &Path, count: usize) -> std::io::Result<()> {
    if !path.exists() {
        let mut file = File::create(path)?;
        let zero_bytes = vec![0u8; count * std::mem::size_of::<Packet>()];
        file.write_all(&zero_bytes)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mmap Read-Write Example ===");

    #[cfg(feature = "mmap")]
    {
        let path = Path::new("data_rw.bin");
        ensure_file(path, 10)?;

        let mut c = raw_bytes::Container::<Packet>::mmap_readwrite(path)?;
        println!("len = {}", c.len());

        // write a value
        c.write(
            0,
            Packet {
                id: 42,
                value: 99.9,
            },
        )?;
        println!("first after write: {:?}", c.get(0)?);
    }

    Ok(())
}
