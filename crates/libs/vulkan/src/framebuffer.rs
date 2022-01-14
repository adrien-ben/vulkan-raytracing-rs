use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{device::VkDevice, VkContext, VkImageView, VkRenderPass};

pub struct VkFramebuffer {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::Framebuffer,
    pub width: u32,
    pub height: u32,
}

impl VkFramebuffer {
    pub(crate) fn new(
        device: Arc<VkDevice>,
        render_pass: &VkRenderPass,
        attachment: &VkImageView,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let attachment = [attachment.inner];
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.inner)
            .attachments(&attachment)
            .width(width)
            .height(height)
            .layers(1);
        let inner = unsafe { device.inner.create_framebuffer(&create_info, None)? };

        Ok(Self {
            device,
            inner,
            width,
            height,
        })
    }
}

impl VkContext {
    pub fn create_framebuffer(
        &self,
        render_pass: &VkRenderPass,
        attachment: &VkImageView,
        width: u32,
        height: u32,
    ) -> Result<VkFramebuffer> {
        VkFramebuffer::new(self.device.clone(), render_pass, attachment, width, height)
    }
}

impl Drop for VkFramebuffer {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_framebuffer(self.inner, None) };
    }
}
