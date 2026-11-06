//! WinSpd adapter for exposing a `BlockStorage` as a raw SCSI disk.
//!
//! Build this module with `--features winspd` and the WinSpd Developer SDK
//! installed. The SDK's user-mode DLL is loaded only when a device starts;
//! normal WinFsp mounts remain usable without WinSpd installed.

use std::ffi::{c_void, OsStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use anyhow::{ensure, Context, Result};
use std::os::windows::ffi::OsStrExt;

use crate::block::BlockStorage;

const ERROR_IO_DEVICE: u32 = 1117;

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCallbacks {
    read: Option<unsafe extern "C" fn(*mut c_void, u64, u32, *mut u8, i32) -> u32>,
    write: Option<unsafe extern "C" fn(*mut c_void, u64, u32, *const u8, i32) -> u32>,
    flush: Option<unsafe extern "C" fn(*mut c_void, u64, u32) -> u32>,
    unmap: Option<unsafe extern "C" fn(*mut c_void, u64, u32) -> u32>,
}

struct CallbackContext {
    storage: Box<dyn BlockStorage>,
}

/// A running WinSpd storage unit. Dropping it shuts down and joins the WinSpd
/// dispatcher before releasing the Rust block-storage callback context.
pub struct WinSpdDevice {
    handle: *mut c_void,
    _context: Box<CallbackContext>,
}

// The native WinSpd object is thread-safe, and its callbacks only touch the
// `Send + Sync` BlockStorage contract. Ownership remains unique in Rust.
unsafe impl Send for WinSpdDevice {}

impl WinSpdDevice {
    /// Start a storage unit. Pass a named pipe such as `\\.\pipe\vramdisk`
    /// for driver-free protocol testing; pass `None` to connect to the
    /// installed WinSpd kernel driver and publish a Windows disk.
    pub fn start(
        storage: Box<dyn BlockStorage>,
        device_name: Option<&OsStr>,
    ) -> Result<Self> {
        let config = storage.config();
        ensure!(config.capacity_bytes > 0, "block-device capacity is empty");
        let mut context = Box::new(CallbackContext { storage });
        let callbacks = NativeCallbacks {
            read: Some(read_callback),
            write: Some(write_callback),
            flush: Some(flush_callback),
            unmap: Some(unmap_callback),
        };
        let wide_name: Vec<u16> = device_name
            .map(|name| name.encode_wide().chain(Some(0)).collect())
            .unwrap_or_default();
        let name_ptr = if wide_name.is_empty() {
            ptr::null_mut()
        } else {
            wide_name.as_ptr() as *mut u16
        };
        let mut handle = ptr::null_mut();
        let error = unsafe {
            vramdisk_winspd_start(
                config.capacity_bytes,
                config.logical_sector_size,
                name_ptr,
                &callbacks,
                (&mut *context as *mut CallbackContext).cast(),
                &mut handle,
            )
        };
        ensure!(error == 0 && !handle.is_null(), "WinSpd could not start storage unit (Windows error {error})");
        Ok(Self {
            handle,
            _context: context,
        })
    }
}

impl Drop for WinSpdDevice {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { vramdisk_winspd_stop(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

unsafe fn callback_context<'a>(context: *mut c_void) -> Result<&'a CallbackContext> {
    (context as *const CallbackContext)
        .as_ref()
        .context("WinSpd callback received a null context")
}

unsafe extern "C" fn read_callback(
    raw_context: *mut c_void,
    lba: u64,
    block_count: u32,
    buffer: *mut u8,
    _flush: i32,
) -> u32 {
    callback_result(|| {
        let context = unsafe { callback_context(raw_context)? };
        let bytes = io_length(context, block_count)?;
        if bytes != 0 && buffer.is_null() {
            anyhow::bail!("WinSpd supplied a null read buffer");
        }
        let output = if bytes == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(buffer, bytes) }
        };
        context.storage.read_blocks(lba, block_count, output)
    })
}

unsafe extern "C" fn write_callback(
    raw_context: *mut c_void,
    lba: u64,
    block_count: u32,
    buffer: *const u8,
    _flush: i32,
) -> u32 {
    callback_result(|| {
        let context = unsafe { callback_context(raw_context)? };
        let bytes = io_length(context, block_count)?;
        if bytes != 0 && buffer.is_null() {
            anyhow::bail!("WinSpd supplied a null write buffer");
        }
        let input = if bytes == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(buffer, bytes) }
        };
        context.storage.write_blocks(lba, block_count, input)
    })
}

unsafe extern "C" fn flush_callback(raw_context: *mut c_void, lba: u64, count: u32) -> u32 {
    callback_result(|| {
        let context = unsafe { callback_context(raw_context)? };
        context.storage.flush_blocks(lba, count)
    })
}

unsafe extern "C" fn unmap_callback(raw_context: *mut c_void, lba: u64, count: u32) -> u32 {
    callback_result(|| {
        let context = unsafe { callback_context(raw_context)? };
        context.storage.unmap_blocks(lba, count)
    })
}

fn io_length(context: &CallbackContext, block_count: u32) -> Result<usize> {
    let bytes = u64::from(block_count)
        .checked_mul(u64::from(context.storage.config().logical_sector_size))
        .context("WinSpd I/O length overflow")?;
    usize::try_from(bytes).context("WinSpd I/O length does not fit this process")
}

fn callback_result(call: impl FnOnce() -> Result<()>) -> u32 {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(())) => 0,
        Ok(Err(_)) | Err(_) => ERROR_IO_DEVICE,
    }
}

/// Start a WinSpd device and keep serving requests until Enter is pressed.
pub fn run_cli(args: &[String]) -> Result<()> {
    use clap::Parser;
    use std::io::{self, Write};

    #[derive(Parser)]
    #[command(name = "vramdisk winspd", no_binary_name = true)]
    struct Args {
        /// pipe runs a protocol test; disk publishes a Windows disk.
        #[arg(long, value_enum, default_value = "pipe")]
        mode: Mode,
        /// VRAM capacity, e.g. 512MiB or 2GiB.
        #[arg(long, value_parser = crate::cli::parse_size)]
        size: u64,
        /// CUDA device ordinal.
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// WinSpd named pipe for stgtest in pipe mode.
        #[arg(long, default_value = DEFAULT_PIPE)]
        pipe: String,
    }

    #[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Pipe,
        Disk,
    }

    let args = Args::try_parse_from(args).context("parse WinSpd command")?;
    let config = crate::block::BlockDeviceConfig::new(
        args.size,
        crate::block::DEFAULT_LOGICAL_SECTOR_SIZE,
    )?;
    let storage = crate::block::GpuBlockStorage::new(args.device, config)?;
    let device_name = match args.mode {
        Mode::Pipe => Some(OsStr::new(&args.pipe)),
        Mode::Disk => None,
    };
    let device = WinSpdDevice::start(Box::new(storage), device_name)?;
    match args.mode {
        Mode::Pipe => {
            println!("WinSpd pipe device is listening at {}\\0", args.pipe);
            println!("Use stgtest against that pipe, then press Enter here to stop.");
        }
        Mode::Disk => {
            println!("WinSpd disk is online. Open Disk Management to initialize and format this new VRAM disk.");
            println!("Only operate on the newly created VRAMDISK device. Press Enter here to stop it.");
        }
    }
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    drop(device);
    Ok(())
}

const DEFAULT_PIPE: &str = r"\\.\pipe\vramdisk";

#[link(name = "winspd_shim", kind = "static")]
extern "C" {
    fn vramdisk_winspd_start(
        capacity_bytes: u64,
        sector_size: u32,
        device_name: *mut u16,
        callbacks: *const NativeCallbacks,
        context: *mut c_void,
        handle: *mut *mut c_void,
    ) -> u32;
    fn vramdisk_winspd_stop(handle: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_rejects_null_context_without_panicking() {
        let error = unsafe { read_callback(ptr::null_mut(), 0, 1, ptr::null_mut(), 0) };
        assert_eq!(error, ERROR_IO_DEVICE);
    }

    #[test]
    fn callback_maps_a_panic_to_io_error() {
        assert_eq!(callback_result(|| panic!("test panic")), ERROR_IO_DEVICE);
    }
}
