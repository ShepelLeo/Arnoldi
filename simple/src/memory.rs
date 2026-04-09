use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TrackingAllocator;

static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);

fn update_peak(next_value: usize) {
    let mut peak = PEAK_BYTES.load(Ordering::SeqCst);

    while next_value > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, next_value, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };

        if !pointer.is_null() {
            let next = CURRENT_BYTES.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            update_peak(next);
        }

        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };

        if !pointer.is_null() {
            let next = CURRENT_BYTES.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            update_peak(next);
        }

        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };

        if !new_pointer.is_null() {
            if new_size >= layout.size() {
                let delta = new_size - layout.size();
                let next = CURRENT_BYTES.fetch_add(delta, Ordering::SeqCst) + delta;
                update_peak(next);
            } else {
                CURRENT_BYTES.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }

        new_pointer
    }
}

pub fn reset_peak() {
    let current = CURRENT_BYTES.load(Ordering::SeqCst);
    BASELINE_BYTES.store(current, Ordering::SeqCst);
    PEAK_BYTES.store(current, Ordering::SeqCst);
}

pub fn peak_bytes_since_reset() -> usize {
    PEAK_BYTES
        .load(Ordering::SeqCst)
        .saturating_sub(BASELINE_BYTES.load(Ordering::SeqCst))
}
