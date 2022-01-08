use std::ffi::CStr;

use anyhow::Result;
use ash::{vk, Instance};

use crate::{queue::VkQueueFamily, surface::VkSurface};

#[derive(Debug, Clone)]
pub struct VkPhysicalDevice {
    pub(crate) inner: vk::PhysicalDevice,
    pub(crate) name: String,
    pub(crate) queue_families: Vec<VkQueueFamily>,
    pub(crate) supported_extensions: Vec<String>,
    pub(crate) supported_surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub(crate) supported_present_modes: Vec<vk::PresentModeKHR>,
}

impl VkPhysicalDevice {
    pub(crate) fn new(
        instance: &Instance,
        surface: &VkSurface,
        inner: vk::PhysicalDevice,
    ) -> Result<Self> {
        let name = unsafe {
            let props = instance.get_physical_device_properties(inner);
            CStr::from_ptr(props.device_name.as_ptr())
                .to_str()
                .unwrap()
                .to_owned()
        };

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(inner) };
        let queue_families = queue_family_properties
            .into_iter()
            .enumerate()
            .map(|(index, f)| {
                let present_support = unsafe {
                    surface.inner.get_physical_device_surface_support(
                        inner,
                        index as _,
                        surface.surface_khr,
                    )?
                };
                Ok(VkQueueFamily::new(index as _, f, present_support))
            })
            .collect::<Result<_>>()?;

        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(inner)? };
        let supported_extensions = extension_properties
            .into_iter()
            .map(|p| {
                let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                name.to_str().unwrap().to_owned()
            })
            .collect();

        let supported_surface_formats = unsafe {
            surface
                .inner
                .get_physical_device_surface_formats(inner, surface.surface_khr)?
        };

        let supported_present_modes = unsafe {
            surface
                .inner
                .get_physical_device_surface_present_modes(inner, surface.surface_khr)?
        };

        Ok(Self {
            inner,
            name,
            queue_families,
            supported_extensions,
            supported_surface_formats,
            supported_present_modes,
        })
    }

    pub fn supports_extensions(&self, extensions: &[&str]) -> bool {
        let supported_extensions = self
            .supported_extensions
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        extensions.iter().all(|e| supported_extensions.contains(e))
    }
}
