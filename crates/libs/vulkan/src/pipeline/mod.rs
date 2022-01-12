mod shader;

pub use shader::*;

use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{device::VkDevice, VkContext, VkDescriptorSetLayout, VkRayTracingContext};

pub struct VkPipelineLayout {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::PipelineLayout,
}

pub struct VkPipeline {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::Pipeline,
}

impl VkPipelineLayout {
    pub(crate) fn new(
        device: Arc<VkDevice>,
        descriptor_set_layouts: &[&VkDescriptorSetLayout],
    ) -> Result<Self> {
        let layouts = descriptor_set_layouts
            .iter()
            .map(|l| l.inner)
            .collect::<Vec<_>>();

        let pipe_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
        let inner = unsafe {
            device
                .inner
                .create_pipeline_layout(&pipe_layout_info, None)?
        };

        Ok(Self { device, inner })
    }
}

impl VkPipeline {
    pub(crate) fn new_ray_tracing(
        device: Arc<VkDevice>,
        ray_tracing: &VkRayTracingContext,
        layout: &VkPipelineLayout,
        create_info: &mut vk::RayTracingPipelineCreateInfoKHR,
    ) -> Result<Self> {
        create_info.layout = layout.inner;

        let inner = unsafe {
            ray_tracing.pipeline_fn.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(create_info),
                None,
            )?[0]
        };

        Ok(Self { device, inner })
    }
}

impl VkContext {
    pub fn create_pipeline_layout(
        &self,
        descriptor_set_layouts: &[&VkDescriptorSetLayout],
    ) -> Result<VkPipelineLayout> {
        VkPipelineLayout::new(self.device.clone(), descriptor_set_layouts)
    }

    pub fn create_ray_tracing_pipeline(
        &self,
        layout: &VkPipelineLayout,
        create_info: &mut vk::RayTracingPipelineCreateInfoKHR,
    ) -> Result<VkPipeline> {
        VkPipeline::new_ray_tracing(self.device.clone(), &self.ray_tracing, layout, create_info)
    }
}

impl Drop for VkPipelineLayout {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_pipeline_layout(self.inner, None) };
    }
}

impl Drop for VkPipeline {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_pipeline(self.inner, None) };
    }
}
