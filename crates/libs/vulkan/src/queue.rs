use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{device::VkDevice, VkCommandBuffer, VkFence, VkSemaphore};

#[derive(Debug, Clone, Copy)]
pub struct VkQueueFamily {
    pub index: u32,
    pub(crate) inner: vk::QueueFamilyProperties,
    supports_present: bool,
}

impl VkQueueFamily {
    pub(crate) fn new(
        index: u32,
        inner: vk::QueueFamilyProperties,
        supports_present: bool,
    ) -> Self {
        Self {
            index,
            inner,
            supports_present,
        }
    }

    pub fn supports_compute(&self) -> bool {
        self.inner.queue_flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn supports_graphics(&self) -> bool {
        self.inner.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn supports_present(&self) -> bool {
        self.supports_present
    }

    pub fn has_queues(&self) -> bool {
        self.inner.queue_count > 0
    }
}

pub struct VkQueue {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::Queue,
}

impl VkQueue {
    pub(crate) fn new(device: Arc<VkDevice>, inner: vk::Queue) -> Self {
        Self { device, inner }
    }

    pub fn submit(
        &self,
        command_buffer: &VkCommandBuffer,
        wait_semaphore: Option<&VkSemaphore>,
        wait_dst_stage_mask: Option<vk::PipelineStageFlags>,
        signal_semaphore: Option<&VkSemaphore>,
        fence: &VkFence,
    ) -> Result<()> {
        let buffs = [command_buffer.inner];
        let wait_semaphores = wait_semaphore.map(|s| vec![s.inner]).unwrap_or_default();
        let wait_dst_stage_mask = wait_dst_stage_mask.map(|f| vec![f]).unwrap_or_default();
        let signal_semaphores = signal_semaphore.map(|s| vec![s.inner]).unwrap_or_default();

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&buffs)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.inner.queue_submit(
                self.inner,
                std::slice::from_ref(&submit_info),
                fence.inner,
            )?;
        };

        Ok(())
    }
}
