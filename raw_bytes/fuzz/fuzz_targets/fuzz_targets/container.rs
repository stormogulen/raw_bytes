#![no_main]

use libfuzzer_sys::fuzz_target;
use raw_bytes::{Container, Storage};
use bytemuck_derive::{Pod, Zeroable};
use bytemuck::cast_slice;

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

    let packets: &[Packet] = cast_slice(data);
    let mut container: Container<Packet> = Container::new_in_memory();

    for &p in packets {
        let _ = container.push(p);
    }

    for i in 0..packets.len() {
        let _ = container.get(i);
        if let Ok(slot) = container.get_mut(i) {
            slot.id ^= 0x12345678;
            slot.value *= 1.001;
        }
    }

    assert_eq!(container.len(), packets.len());
});
