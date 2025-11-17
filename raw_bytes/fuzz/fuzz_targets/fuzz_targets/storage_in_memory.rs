#![no_main]

use libfuzzer_sys::fuzz_target;
use raw_bytes::Storage;
use bytemuck_derive::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, Debug, PartialEq)]
struct Packet {
    id: u32,
    value: f32,
}

fuzz_target!(|data: &[u8]| {
    if data.len() % std::mem::size_of::<Packet>() != 0 {
        return;
    }

    let packets: &[Packet] = bytemuck::cast_slice(data);

    let mut storage = Storage::<Packet>::new_in_memory();

    for &p in packets {
        let _ = storage.push(p);
    }

    for (i, _) in packets.iter().enumerate() {
        let _ = storage.get(i);
    }

    for i in 0..packets.len() {
        if let Ok(slot) = storage.get_mut(i) {
            slot.id ^= 0xDEADBEEF;
            slot.value *= 1.01;
        }
    }

    assert_eq!(storage.len(), packets.len());
});
