use std::sync::Arc;

use anyhow::Result;
use ash::{vk, Device};

use crate::{
    instance::VkInstance,
    physical_device::VkPhysicalDevice,
    queue::{VkQueue, VkQueueFamily},
};

pub struct VkDevice {
    pub(crate) inner: Device,
}

impl VkDevice {
    pub(crate) fn new(
        instance: &VkInstance,
        physical_device: &VkPhysicalDevice,
        queue_families: &[VkQueueFamily],
    ) -> Result<Self> {
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            let mut indices = queue_families.iter().map(|f| f.index).collect::<Vec<_>>();
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<_>>()
        };

        // TODO: pass in as args
        let device_extensions_ptrs = [
            ash::extensions::khr::Swapchain::name().as_ptr(),
            ash::extensions::khr::RayTracingPipeline::name().as_ptr(),
            ash::extensions::khr::AccelerationStructure::name().as_ptr(),
            ash::extensions::khr::DeferredHostOperations::name().as_ptr(),
        ];

        let mut ray_tracing_feature =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);
        let mut acceleration_struct_feature =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true);
        let mut device_addr_feature =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::builder().buffer_device_address(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .runtime_descriptor_array(true)
            .buffer_device_address(true);

        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .features(vk::PhysicalDeviceFeatures::default())
            .push_next(&mut device_addr_feature)
            .push_next(&mut acceleration_struct_feature)
            .push_next(&mut ray_tracing_feature)
            .push_next(&mut vulkan_12_features);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .push_next(&mut features);

        let inner = unsafe {
            instance
                .inner
                .create_device(physical_device.inner, &device_create_info, None)?
        };

        Ok(Self { inner })
    }

    pub fn get_queue(self: &Arc<Self>, queue_family: VkQueueFamily, queue_index: u32) -> VkQueue {
        let inner = unsafe { self.inner.get_device_queue(queue_family.index, queue_index) };
        VkQueue::new(self.clone(), inner)
    }
}

impl Drop for VkDevice {
    fn drop(&mut self) {
        unsafe {
            self.inner.destroy_device(None);
        }
    }
}
