use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{device::VkDevice, VkContext};

pub struct VkRenderPass {
    device: Arc<VkDevice>,
    pub inner: vk::RenderPass,
}

impl VkRenderPass {
    pub(crate) fn new(
        device: Arc<VkDevice>,
        create_info: &vk::RenderPassCreateInfo,
    ) -> Result<Self> {
        let inner = unsafe { device.inner.create_render_pass(create_info, None)? };

        Ok(Self { device, inner })
    }
}

impl VkContext {
    pub fn create_render_pass(
        &self,
        create_info: &vk::RenderPassCreateInfo,
    ) -> Result<VkRenderPass> {
        VkRenderPass::new(self.device.clone(), create_info)
    }
}

impl Drop for VkRenderPass {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_render_pass(self.inner, None) };
    }
}
