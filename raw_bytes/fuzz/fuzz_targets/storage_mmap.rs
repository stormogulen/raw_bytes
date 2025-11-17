#![no_main]

#[cfg(feature = "mmap")]
use libfuzzer_sys::fuzz_target;
#[cfg(feature = "mmap")]
use raw_bytes::Storage;
#[cfg(feature = "mmap")]
use bytemuck_derive::{Pod, Zeroable};
#[cfg(feature = "mmap")]
use tempfile::NamedTempFile;
#[cfg(feature = "mmap")]
use std::io::Write;
#[cfg(feature = "mmap")]
use bytemuck::cast_slice;

#[cfg(feature = "mmap")]
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, Debug, PartialEq)]
struct Packet {
    id: u32,
    value: f32,
}

#[cfg(feature = "mmap")]
fuzz_target!(|data: &[u8]| {
    if data.len() % std::mem::size_of::<Packet>() != 0 || data.is_empty() {
        return;
    }

    let packets: &[Packet] = cast_slice(data);

    // write packets to temp file
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(bytemuck::cast_slice(packets)).unwrap();
    file.flush().unwrap();

    // read-only mmap
    let storage = Storage::<Packet>::from_mmap_readonly(file.path()).unwrap();
    for i in 0..packets.len() {
        let _ = storage.get(i);
    }

    // read-write mmap
    let mut storage_rw = Storage::<Packet>::from_mmap_readwrite(file.path()).unwrap();
    for i in 0..packets.len() {
        if let Ok(slot) = storage_rw.get_mut(i) {
            slot.id ^= 0xCAFEBABE;
            slot.value *= 0.99;
        }
    }
});
