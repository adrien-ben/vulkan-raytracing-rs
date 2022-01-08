mod buffer;
mod command;
mod context;
mod descriptor;
mod device;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod queue;
mod ray_tracing;
mod surface;
mod swapchain;
mod sync;

pub mod utils;

pub use buffer::*;
pub use command::*;
pub use context::*;
pub use descriptor::*;
pub use image::*;
pub use pipeline::*;
pub use queue::*;
pub use ray_tracing::*;
pub use swapchain::*;
pub use sync::*;

#[derive(Debug, Clone, Copy, Default)]
pub struct VkVersion {
    pub variant: u32,
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl VkVersion {
    pub fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        Self {
            variant,
            major,
            minor,
            patch,
        }
    }

    pub fn from_major(major: u32) -> Self {
        Self {
            major,
            ..Default::default()
        }
    }

    pub fn from_major_minor(major: u32, minor: u32) -> Self {
        Self {
            major,
            minor,
            ..Default::default()
        }
    }

    pub(crate) fn make_api_version(&self) -> u32 {
        ash::vk::make_api_version(self.variant, self.major, self.minor, self.patch)
    }
}
