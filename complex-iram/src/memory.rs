use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TrackingAllocator;

static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);

static DEVICE_CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEVICE_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEVICE_BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEVICE_ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_TO_DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEVICE_TO_HOST_BYTES: AtomicUsize = AtomicUsize::new(0);

fn update_peak(next_value: usize) {
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);

    while next_value > peak {
        match PEAK_BYTES.compare_exchange_weak(
            peak,
            next_value,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

fn update_atomic_peak(peak_counter: &AtomicUsize, next_value: usize) {
    let mut peak = peak_counter.load(Ordering::Relaxed);

    while next_value > peak {
        match peak_counter.compare_exchange_weak(
            peak,
            next_value,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };

        if !pointer.is_null() {
            let next = CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            update_peak(next);
        }

        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };

        if !pointer.is_null() {
            let next = CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            update_peak(next);
        }

        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };

        if !new_pointer.is_null() {
            if new_size >= layout.size() {
                let delta = new_size - layout.size();
                let next = CURRENT_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
                update_peak(next);
            } else {
                CURRENT_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }

        new_pointer
    }
}

pub fn reset_peak() {
    let current = CURRENT_BYTES.load(Ordering::Relaxed);
    BASELINE_BYTES.store(current, Ordering::Relaxed);
    PEAK_BYTES.store(current, Ordering::Relaxed);

    let device_current = DEVICE_CURRENT_BYTES.load(Ordering::Relaxed);
    DEVICE_BASELINE_BYTES.store(device_current, Ordering::Relaxed);
    DEVICE_PEAK_BYTES.store(device_current, Ordering::Relaxed);
    DEVICE_ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    HOST_TO_DEVICE_BYTES.store(0, Ordering::Relaxed);
    DEVICE_TO_HOST_BYTES.store(0, Ordering::Relaxed);
}

pub fn peak_bytes_since_reset() -> usize {
    PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(BASELINE_BYTES.load(Ordering::Relaxed))
}

pub fn record_device_allocation(bytes: usize) {
    if bytes == 0 {
        return;
    }
    let next = DEVICE_CURRENT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    DEVICE_ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    update_atomic_peak(&DEVICE_PEAK_BYTES, next);
}

pub fn record_device_deallocation(bytes: usize) {
    if bytes == 0 {
        return;
    }
    DEVICE_CURRENT_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

pub fn record_host_to_device_copy(bytes: usize) {
    HOST_TO_DEVICE_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn record_device_to_host_copy(bytes: usize) {
    DEVICE_TO_HOST_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn peak_device_bytes_since_reset() -> usize {
    DEVICE_PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(DEVICE_BASELINE_BYTES.load(Ordering::Relaxed))
}

pub fn device_allocation_count_since_reset() -> usize {
    DEVICE_ALLOCATION_COUNT.load(Ordering::Relaxed)
}

pub fn host_to_device_bytes_since_reset() -> usize {
    HOST_TO_DEVICE_BYTES.load(Ordering::Relaxed)
}

pub fn device_to_host_bytes_since_reset() -> usize {
    DEVICE_TO_HOST_BYTES.load(Ordering::Relaxed)
}
