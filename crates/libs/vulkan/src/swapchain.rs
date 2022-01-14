use std::sync::Arc;

use anyhow::Result;
use ash::{extensions::khr::Swapchain, vk};

use crate::{
    device::VkDevice, VkContext, VkFramebuffer, VkImage, VkImageView, VkQueue, VkRenderPass,
    VkSemaphore,
};

pub struct AcquiredImage {
    pub index: u32,
    pub is_suboptimal: bool,
}

pub struct VkSwapchain {
    device: Arc<VkDevice>,
    inner: Swapchain,
    swapchain_khr: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<VkImage>,
    pub views: Vec<VkImageView>,
}

impl VkSwapchain {
    pub fn new(context: &VkContext, width: u32, height: u32) -> Result<Self> {
        log::debug!("Creating vulkan swapchain");

        let device = context.device.clone();

        // Swapchain format
        let format = {
            let formats = unsafe {
                context.surface.inner.get_physical_device_surface_formats(
                    context.physical_device.inner,
                    context.surface.surface_khr,
                )?
            };
            if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_UNORM,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                }
            } else {
                *formats
                    .iter()
                    .find(|format| {
                        format.format == vk::Format::B8G8R8A8_UNORM
                            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .unwrap_or(&formats[0])
            }
        };
        log::debug!("Swapchain format: {format:?}");

        // Swapchain present mode
        let present_mode = {
            let present_modes = unsafe {
                context
                    .surface
                    .inner
                    .get_physical_device_surface_present_modes(
                        context.physical_device.inner,
                        context.surface.surface_khr,
                    )
                    .expect("Failed to get physical device surface present modes")
            };
            if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            }
        };
        log::debug!("Swapchain present mode: {present_mode:?}");

        let capabilities = unsafe {
            context
                .surface
                .inner
                .get_physical_device_surface_capabilities(
                    context.physical_device.inner,
                    context.surface.surface_khr,
                )?
        };

        // Swapchain extent
        let extent = {
            if capabilities.current_extent.width != std::u32::MAX {
                capabilities.current_extent
            } else {
                let min = capabilities.min_image_extent;
                let max = capabilities.max_image_extent;
                let width = width.min(max.width).max(min.width);
                let height = height.min(max.height).max(min.height);
                vk::Extent2D { width, height }
            }
        };
        log::debug!("Swapchain extent: {extent:?}");

        // Swapchain image count
        let image_count = capabilities.min_image_count;
        log::debug!("Swapchain image count: {image_count:?}");

        // Swapchain
        let families_indices = [
            context.graphics_queue_family.index,
            context.present_queue_family.index,
        ];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(context.surface.surface_khr)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                );

            builder = if context.graphics_queue_family.index != context.present_queue_family.index {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let inner = Swapchain::new(&context.instance.inner, &context.device.inner);
        let swapchain_khr = unsafe { inner.create_swapchain(&create_info, None)? };

        // Swapchain images and image views
        let images = unsafe { inner.get_swapchain_images(swapchain_khr)? };
        let images = images
            .into_iter()
            .map(|i| {
                VkImage::from_swapchain_image(
                    device.clone(),
                    context.allocator.clone(),
                    i,
                    format.format,
                    extent,
                )
            })
            .collect::<Vec<_>>();

        let views = images
            .iter()
            .map(VkImage::create_image_view)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            device,
            inner,
            swapchain_khr,
            extent,
            format: format.format,
            color_space: format.color_space,
            present_mode,
            images,
            views,
        })
    }

    pub fn resize(&mut self, context: &VkContext, width: u32, height: u32) -> Result<()> {
        log::debug!("Resizing vulkan swapchain to {width}x{height}");

        self.destroy();

        let capabilities = unsafe {
            context
                .surface
                .inner
                .get_physical_device_surface_capabilities(
                    context.physical_device.inner,
                    context.surface.surface_khr,
                )?
        };

        // Swapchain extent
        let extent = {
            if capabilities.current_extent.width != std::u32::MAX {
                capabilities.current_extent
            } else {
                let min = capabilities.min_image_extent;
                let max = capabilities.max_image_extent;
                let width = width.min(max.width).max(min.width);
                let height = height.min(max.height).max(min.height);
                vk::Extent2D { width, height }
            }
        };
        log::debug!("Swapchain extent: {extent:?}");

        // Swapchain image count
        let image_count = capabilities.min_image_count;

        // Swapchain
        let families_indices = [
            context.graphics_queue_family.index,
            context.present_queue_family.index,
        ];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(context.surface.surface_khr)
                .min_image_count(image_count)
                .image_format(self.format)
                .image_color_space(self.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                );

            builder = if context.graphics_queue_family.index != context.present_queue_family.index {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(self.present_mode)
                .clipped(true)
        };

        let swapchain_khr = unsafe { self.inner.create_swapchain(&create_info, None)? };

        // Swapchain images and image views
        let images = unsafe { self.inner.get_swapchain_images(swapchain_khr)? };
        let images = images
            .into_iter()
            .map(|i| {
                VkImage::from_swapchain_image(
                    self.device.clone(),
                    context.allocator.clone(),
                    i,
                    self.format,
                    extent,
                )
            })
            .collect::<Vec<_>>();

        let views = images
            .iter()
            .map(VkImage::create_image_view)
            .collect::<Result<Vec<_>, _>>()?;

        self.swapchain_khr = swapchain_khr;
        self.extent = extent;
        self.images = images;
        self.views = views;

        Ok(())
    }

    pub fn acquire_next_image(
        &self,
        timeout: u64,
        semaphore: &VkSemaphore,
    ) -> Result<AcquiredImage> {
        let (index, is_suboptimal) = unsafe {
            self.inner.acquire_next_image(
                self.swapchain_khr,
                timeout,
                semaphore.inner,
                vk::Fence::null(),
            )?
        };

        Ok(AcquiredImage {
            index,
            is_suboptimal,
        })
    }

    pub fn queue_present(
        &self,
        image_index: u32,
        wait_semaphores: &[&VkSemaphore],
        queue: &VkQueue,
    ) -> Result<bool> {
        let swapchains = [self.swapchain_khr];
        let images_indices = [image_index];
        let wait_semaphores = wait_semaphores.iter().map(|s| s.inner).collect::<Vec<_>>();

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&images_indices);

        let result = unsafe { self.inner.queue_present(queue.inner, &present_info)? };

        Ok(result)
    }

    pub fn get_framebuffers(&self, render_pass: &VkRenderPass) -> Result<Vec<VkFramebuffer>> {
        self.views
            .iter()
            .map(|view| {
                VkFramebuffer::new(
                    self.device.clone(),
                    render_pass,
                    view,
                    self.extent.width,
                    self.extent.height,
                )
            })
            .collect::<Result<Vec<_>>>()
    }

    fn destroy(&mut self) {
        unsafe {
            self.views.clear();
            self.images.clear();
            self.inner.destroy_swapchain(self.swapchain_khr, None);
        }
    }
}

impl Drop for VkSwapchain {
    fn drop(&mut self) {
        self.destroy();
    }
}
