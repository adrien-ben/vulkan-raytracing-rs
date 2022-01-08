use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{device::VkDevice, VkAccelerationStructure, VkContext, VkImageView};

pub struct VkDescriptorSetLayout {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::DescriptorSetLayout,
}

impl VkDescriptorSetLayout {
    pub(crate) fn new(
        device: Arc<VkDevice>,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<Self> {
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        let inner = unsafe { device.inner.create_descriptor_set_layout(&dsl_info, None)? };

        Ok(Self { device, inner })
    }
}

impl Drop for VkDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .inner
                .destroy_descriptor_set_layout(self.inner, None);
        }
    }
}

pub struct VkDescriptorPool {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::DescriptorPool,
}

impl VkDescriptorPool {
    pub(crate) fn new(
        device: Arc<VkDevice>,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);
        let inner = unsafe { device.inner.create_descriptor_pool(&pool_info, None)? };

        Ok(Self { device, inner })
    }

    pub fn allocate_sets(
        &self,
        layout: &VkDescriptorSetLayout,
        count: u32,
    ) -> Result<VkDescriptorSets> {
        let layouts = (0..count).map(|_| layout.inner).collect::<Vec<_>>();
        let sets_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.inner)
            .set_layouts(&layouts);
        let sets = unsafe {
            self.device
                .inner
                .allocate_descriptor_sets(&sets_alloc_info)?
        };
        let sets = sets
            .into_iter()
            .map(|inner| VkDescriptorSet {
                device: self.device.clone(),
                inner,
            })
            .collect::<Vec<_>>();

        Ok(VkDescriptorSets { sets })
    }
}

impl Drop for VkDescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_descriptor_pool(self.inner, None) };
    }
}

pub struct VkDescriptorSet {
    device: Arc<VkDevice>,
    pub(crate) inner: vk::DescriptorSet,
}

impl VkDescriptorSet {
    pub fn update(&self, write: &VkWriteDescriptorSet) {
        use VkWriteDescriptorSetKind::*;
        match write.kind {
            StorageImage { view, layout } => {
                let img_info = vk::DescriptorImageInfo::builder()
                    .image_view(view.inner)
                    .image_layout(layout);
                let write = vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_binding(write.binding)
                    .dst_set(self.inner)
                    .image_info(std::slice::from_ref(&img_info));

                unsafe {
                    self.device
                        .inner
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
            }
            AccelerationStructure {
                acceleration_structure,
            } => {
                let mut write_set_as = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                    .acceleration_structures(std::slice::from_ref(&acceleration_structure.inner));

                let mut write = vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .dst_binding(write.binding)
                    .dst_set(self.inner)
                    .push_next(&mut write_set_as)
                    .build();
                write.descriptor_count = 1;

                unsafe {
                    self.device
                        .inner
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
            }
        }
    }
}

pub struct VkDescriptorSets {
    pub sets: Vec<VkDescriptorSet>,
}

impl VkDescriptorSets {
    pub fn iter(&self) -> std::slice::Iter<VkDescriptorSet> {
        self.sets.iter()
    }
}

impl VkContext {
    pub fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<VkDescriptorSetLayout> {
        VkDescriptorSetLayout::new(self.device.clone(), bindings)
    }

    pub fn create_descriptor_pool(
        &self,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<VkDescriptorPool> {
        VkDescriptorPool::new(self.device.clone(), max_sets, pool_sizes)
    }
}

pub struct VkWriteDescriptorSet<'a> {
    pub binding: u32,
    pub kind: VkWriteDescriptorSetKind<'a>,
}
pub enum VkWriteDescriptorSetKind<'a> {
    StorageImage {
        view: &'a VkImageView,
        layout: vk::ImageLayout,
    },
    AccelerationStructure {
        acceleration_structure: &'a VkAccelerationStructure,
    },
}
