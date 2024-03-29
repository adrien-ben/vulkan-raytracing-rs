use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{
    device::VkDevice, VkBuffer, VkContext, VkDescriptorSet, VkImage, VkImageView, VkPipelineLayout,
    VkQueueFamily, VkRTPipeline, VkRayTracingContext, VkShaderBindingTable,
};

pub struct VkCommandPool {
    device: Arc<VkDevice>,
    ray_tracing: Arc<VkRayTracingContext>,
    pub inner: vk::CommandPool,
}

impl VkCommandPool {
    pub(crate) fn new(
        device: Arc<VkDevice>,
        ray_tracing: Arc<VkRayTracingContext>,
        queue_family: VkQueueFamily,
        flags: Option<vk::CommandPoolCreateFlags>,
    ) -> Result<Self> {
        let flags = flags.unwrap_or_else(vk::CommandPoolCreateFlags::empty);

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family.index)
            .flags(flags);
        let inner = unsafe { device.inner.create_command_pool(&command_pool_info, None)? };

        Ok(Self {
            device,
            ray_tracing,
            inner,
        })
    }

    pub fn allocate_command_buffers(
        &self,
        level: vk::CommandBufferLevel,
        count: u32,
    ) -> Result<Vec<VkCommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.inner)
            .level(level)
            .command_buffer_count(count);

        let buffers = unsafe { self.device.inner.allocate_command_buffers(&allocate_info)? };
        let buffers = buffers
            .into_iter()
            .map(|inner| VkCommandBuffer {
                device: self.device.clone(),
                ray_tracing: self.ray_tracing.clone(),
                inner,
            })
            .collect();

        Ok(buffers)
    }

    pub fn allocate_command_buffer(
        &self,
        level: vk::CommandBufferLevel,
    ) -> Result<VkCommandBuffer> {
        let buffers = self.allocate_command_buffers(level, 1)?;
        let buffer = buffers.into_iter().next().unwrap();

        Ok(buffer)
    }

    pub fn free_command_buffers(&self, buffer: &[VkCommandBuffer]) {
        let buffs = buffer.iter().map(|b| b.inner).collect::<Vec<_>>();
        unsafe { self.device.inner.free_command_buffers(self.inner, &buffs) };
    }

    pub fn free_command_buffer(&self, buffer: &VkCommandBuffer) -> Result<()> {
        let buffs = [buffer.inner];
        unsafe { self.device.inner.free_command_buffers(self.inner, &buffs) };

        Ok(())
    }
}

impl VkContext {
    pub fn create_command_pool(
        &self,
        queue_family: VkQueueFamily,
        flags: Option<vk::CommandPoolCreateFlags>,
    ) -> Result<VkCommandPool> {
        VkCommandPool::new(
            self.device.clone(),
            self.ray_tracing.clone(),
            queue_family,
            flags,
        )
    }
}

impl Drop for VkCommandPool {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_command_pool(self.inner, None) };
    }
}

pub struct VkCommandBuffer {
    device: Arc<VkDevice>,
    ray_tracing: Arc<VkRayTracingContext>,
    pub inner: vk::CommandBuffer,
}

impl VkCommandBuffer {
    pub fn begin(&self, flags: Option<vk::CommandBufferUsageFlags>) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(flags.unwrap_or(vk::CommandBufferUsageFlags::empty()));
        unsafe {
            self.device
                .inner
                .begin_command_buffer(self.inner, &begin_info)?
        };

        Ok(())
    }

    pub fn end(&self) -> Result<()> {
        unsafe { self.device.inner.end_command_buffer(self.inner)? };

        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.device
                .inner
                .reset_command_buffer(self.inner, vk::CommandBufferResetFlags::empty())?
        };

        Ok(())
    }

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: &VkRTPipeline) {
        unsafe {
            self.device
                .inner
                .cmd_bind_pipeline(self.inner, bind_point, pipeline.inner)
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: &VkPipelineLayout,
        first_set: u32,
        sets: &[&VkDescriptorSet],
    ) {
        let sets = sets.iter().map(|s| s.inner).collect::<Vec<_>>();
        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                self.inner,
                bind_point,
                layout.inner,
                first_set,
                &sets,
                &[],
            )
        }
    }

    pub fn copy_buffer(&self, src_buffer: &VkBuffer, dst_buffer: &VkBuffer) {
        unsafe {
            let region = vk::BufferCopy::builder().size(src_buffer.size);
            self.device.inner.cmd_copy_buffer(
                self.inner,
                src_buffer.inner,
                dst_buffer.inner,
                std::slice::from_ref(&region),
            )
        };
    }

    pub fn pipeline_image_barriers(&self, barriers: &[VkImageBarrier]) {
        let barriers = barriers
            .iter()
            .map(|b| {
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(b.src_stage_mask)
                    .src_access_mask(b.src_access_mask)
                    .old_layout(b.old_layout)
                    .dst_stage_mask(b.dst_stage_mask)
                    .dst_access_mask(b.dst_access_mask)
                    .new_layout(b.new_layout)
                    .image(b.image.inner)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build()
            })
            .collect::<Vec<_>>();

        let dependency_info = vk::DependencyInfo::builder().image_memory_barriers(&barriers);

        unsafe {
            self.device
                .inner
                .cmd_pipeline_barrier2(self.inner, &dependency_info)
        };
    }

    pub fn copy_image(
        &self,
        src_image: &VkImage,
        src_layout: vk::ImageLayout,
        dst_image: &VkImage,
        dst_layout: vk::ImageLayout,
    ) {
        let region = vk::ImageCopy::builder()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                mip_level: 0,
                layer_count: 1,
            })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                mip_level: 0,
                layer_count: 1,
            })
            .extent(vk::Extent3D {
                width: src_image.extent.width,
                height: src_image.extent.height,
                depth: 1,
            });

        unsafe {
            self.device.inner.cmd_copy_image(
                self.inner,
                src_image.inner,
                src_layout,
                dst_image.inner,
                dst_layout,
                std::slice::from_ref(&region),
            )
        };
    }

    pub fn copy_buffer_to_image(&self, src: &VkBuffer, dst: &VkImage, layout: vk::ImageLayout) {
        let region = vk::BufferImageCopy::builder()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(dst.extent);

        unsafe {
            self.device.inner.cmd_copy_buffer_to_image(
                self.inner,
                src.inner,
                dst.inner,
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    pub fn build_acceleration_structures(
        &self,
        as_build_geo_info: &vk::AccelerationStructureBuildGeometryInfoKHR,
        as_build_range_info: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        unsafe {
            self.ray_tracing
                .acceleration_structure_fn
                .cmd_build_acceleration_structures(
                    self.inner,
                    std::slice::from_ref(as_build_geo_info),
                    std::slice::from_ref(&as_build_range_info),
                )
        };
    }

    pub fn trace_rays(&self, shader_binding_table: &VkShaderBindingTable, width: u32, height: u32) {
        let empty_region = vk::StridedDeviceAddressRegionKHR::builder();
        unsafe {
            self.ray_tracing.pipeline_fn.cmd_trace_rays(
                self.inner,
                &shader_binding_table.raygen_region,
                &shader_binding_table.miss_region,
                &shader_binding_table.hit_region,
                &empty_region,
                width,
                height,
                1,
            )
        };
    }

    pub fn begin_rendering(&self, image_view: &VkImageView, extent: vk::Extent2D) {
        let color_attachment_info = vk::RenderingAttachmentInfo::builder()
            .image_view(image_view.inner)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            });

        let rendering_info = vk::RenderingInfo::builder()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        unsafe {
            self.device
                .inner
                .cmd_begin_rendering(self.inner, &rendering_info)
        };
    }

    pub fn end_rendering(&self) {
        unsafe { self.device.inner.cmd_end_rendering(self.inner) };
    }
}

pub struct VkImageBarrier<'a> {
    pub image: &'a VkImage,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_access_mask: vk::AccessFlags2,
    pub dst_access_mask: vk::AccessFlags2,
    pub src_stage_mask: vk::PipelineStageFlags2,
    pub dst_stage_mask: vk::PipelineStageFlags2,
}
