use anyhow::Result;
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{
            AccelerationStructure as AccelerationStructureFn, DeferredHostOperations,
            RayTracingPipeline, Surface, Swapchain as SwapchainFn,
        },
    },
    vk::{self, Packed24_8},
    Device, Entry, Instance,
};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings, MemoryLocation,
};
use simple_logger::SimpleLogger;
use std::{
    ffi::{CStr, CString},
    mem::{align_of, size_of, size_of_val},
    os::raw::c_void,
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 576;
const APP_NAME: &str = "Triangle basic";

fn main() -> Result<()> {
    SimpleLogger::default().env().init()?;

    let (window, event_loop) = create_window();
    let mut app = App::new(&window)?;
    let mut is_swapchain_dirty = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            // On resize
            Event::WindowEvent {
                event: WindowEvent::Resized(..),
                ..
            } => {
                log::debug!("Window has been resized");
                is_swapchain_dirty = true;
            }
            // Draw
            Event::MainEventsCleared => {
                if is_swapchain_dirty {
                    let dim = window.inner_size();
                    if dim.width > 0 && dim.height > 0 {
                        app.recreate_swapchain().expect("Failed to recreate swap");
                    } else {
                        return;
                    }
                }

                is_swapchain_dirty = app.draw().expect("Failed to tick");
            }
            // Exit app on request to close window
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            // Wait for gpu to finish pending work before closing app
            Event::LoopDestroyed => app
                .wait_for_gpu()
                .expect("Failed to wait for gpu to finish work"),
            _ => (),
        }
    });
}

struct Swapchain {
    swapchain_fn: SwapchainFn,
    swapchain_khr: vk::SwapchainKHR,
    extent: vk::Extent2D,
    format: vk::Format,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

impl Swapchain {
    fn destroy(&mut self, device: &Device) -> Result<()> {
        unsafe {
            self.views
                .iter()
                .for_each(|v| device.destroy_image_view(*v, None));
            self.views.clear();
            self.swapchain_fn
                .destroy_swapchain(self.swapchain_khr, None);
        }

        Ok(())
    }
}

struct Buffer {
    handle: vk::Buffer,
    allocation: Option<Allocation>,
}

impl Buffer {
    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        unsafe { device.destroy_buffer(self.handle, None) };
        allocator.free(self.allocation.take().unwrap())?;

        Ok(())
    }
}

struct ImageAndView {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl ImageAndView {
    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        allocator.free(self.allocation.take().unwrap())?;

        Ok(())
    }
}

struct AccelerationStructure {
    handle: vk::AccelerationStructureKHR,
    buffer: Buffer,
    address: u64,
}

impl AccelerationStructure {
    fn destroy(
        &mut self,
        device: &Device,
        allocator: &mut Allocator,
        acceleration_struct_fn: &AccelerationStructureFn,
    ) -> Result<()> {
        unsafe {
            acceleration_struct_fn.destroy_acceleration_structure(self.handle, None);
        }
        self.buffer.destroy(device, allocator)?;

        Ok(())
    }
}

struct Sbt {
    buffer: Buffer,
    _address: u64,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl Sbt {
    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        self.buffer.destroy(device, allocator)?;
        Ok(())
    }
}

struct App {
    _entry: Entry,
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    surface: Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    allocator: Option<Allocator>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    command_pool: vk::CommandPool,
    swapchain: Swapchain,
    _rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    rt_pipeline_fn: RayTracingPipeline,
    _acceleration_struct_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    acceleration_struct_fn: AccelerationStructureFn,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bottom_as: AccelerationStructure,
    instance_buffer: Buffer,
    top_as: AccelerationStructure,
    storage_image: ImageAndView,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sbt: Sbt,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl App {
    fn new(window: &Window) -> Result<Self> {
        log::info!("Create application");

        // Vulkan instance
        let entry = Entry::linked();
        let (instance, debug_utils, debug_utils_messenger) =
            create_vulkan_instance(&entry, window)?;

        // Vulkan surface
        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe { ash_window::create_surface(&entry, &instance, window, None)? };

        // Vulkan physical device and queue families indices (graphics and present)
        let (physical_device, graphics_q_index, present_q_index) =
            create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
                &instance,
                &surface,
                surface_khr,
            )?;

        // Vulkan logical device and queues
        let (device, graphics_queue, present_queue) =
            create_vulkan_device_and_graphics_and_present_qs(
                &instance,
                physical_device,
                graphics_q_index,
                present_q_index,
            )?;

        // Gpu allocator
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings {
                log_allocations: true,
                log_frees: true,
                ..Default::default()
            },
            buffer_device_address: true,
        })?;

        // Command pool
        let command_pool = {
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_q_index)
                .flags(vk::CommandPoolCreateFlags::empty());
            unsafe { device.create_command_pool(&command_pool_info, None)? }
        };

        // Swapchain
        let swapchain = create_vulkan_swapchain(
            &instance,
            &surface,
            surface_khr,
            physical_device,
            graphics_q_index,
            present_q_index,
            &device,
        )?;

        // Ray tracing init
        let rt_pipeline_properties =
            unsafe { RayTracingPipeline::get_properties(&instance, physical_device) };
        log::debug!(
            "Ray tracing pipeline properties {:#?}",
            rt_pipeline_properties
        );
        let rt_pipeline_fn = RayTracingPipeline::new(&instance, &device);

        let acceleration_struct_properties =
            unsafe { AccelerationStructureFn::get_properties(&instance, physical_device) };
        log::debug!(
            "Acceleration structure properties {:#?}",
            acceleration_struct_properties
        );
        let acceleration_struct_fn = AccelerationStructureFn::new(&instance, &device);

        // Bottom AS
        let (bottom_as, vertex_buffer, index_buffer) = create_bottom_as(
            &device,
            &mut allocator,
            &acceleration_struct_fn,
            graphics_queue,
            command_pool,
        )?;

        // Top AS
        let (top_as, instance_buffer) = create_top_as(
            &device,
            &mut allocator,
            &acceleration_struct_fn,
            &bottom_as,
            graphics_queue,
            command_pool,
        )?;

        // Storage image
        let storage_image = create_storage_image(
            &device,
            &mut allocator,
            swapchain.format,
            swapchain.extent,
            graphics_queue,
            command_pool,
        )?;

        // RT pipeline
        let (descriptor_set_layout, pipeline_layout, pipeline) =
            create_pipeline(&device, &rt_pipeline_fn)?;

        // Shader Binding Table (SBT)
        let sbt = create_shader_binding_table(
            &device,
            &mut allocator,
            &rt_pipeline_fn,
            rt_pipeline_properties,
            pipeline,
        )?;

        // RT Descriptor sets
        let (descriptor_pool, descriptor_set) =
            create_descriptor_set(&device, descriptor_set_layout, &top_as, &storage_image)?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &device,
            command_pool,
            &swapchain,
            &rt_pipeline_fn,
            &sbt,
            pipeline_layout,
            pipeline,
            descriptor_set,
            &storage_image,
        )?;

        // Semaphore use for presentation
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            unsafe { device.create_semaphore(&semaphore_info, None)? }
        };
        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            unsafe { device.create_semaphore(&semaphore_info, None)? }
        };
        let fence = {
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            unsafe { device.create_fence(&fence_info, None)? }
        };

        Ok(Self {
            _entry: entry,
            instance,
            debug_utils,
            debug_utils_messenger,
            surface,
            surface_khr,
            physical_device,
            graphics_q_index,
            present_q_index,
            device,
            graphics_queue,
            present_queue,
            allocator: Some(allocator),
            vertex_buffer,
            index_buffer,
            command_pool,
            swapchain,
            _rt_pipeline_properties: rt_pipeline_properties,
            rt_pipeline_fn,
            _acceleration_struct_properties: acceleration_struct_properties,
            acceleration_struct_fn,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            bottom_as,
            instance_buffer,
            top_as,
            storage_image,
            descriptor_pool,
            descriptor_set,
            sbt,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            fence,
        })
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        log::debug!("Recreating the swapchain");

        self.wait_for_gpu()?;

        unsafe { self.cleanup_swapchain() };

        // Swapchain
        let swapchain = create_vulkan_swapchain(
            &self.instance,
            &self.surface,
            self.surface_khr,
            self.physical_device,
            self.graphics_q_index,
            self.present_q_index,
            &self.device,
        )?;

        // Recreate storage image for RT and update descriptor set
        let storage_image = create_storage_image(
            &self.device,
            self.allocator.as_mut().unwrap(),
            swapchain.format,
            swapchain.extent,
            self.graphics_queue,
            self.command_pool,
        )?;

        let desc_img_info = vk::DescriptorImageInfo::builder()
            .image_view(storage_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let img_write_set = vk::WriteDescriptorSet::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .image_info(std::slice::from_ref(&desc_img_info))
            .build();

        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&img_write_set), &[])
        };

        let mut old_img_storage = std::mem::replace(&mut self.storage_image, storage_image);
        old_img_storage.destroy(&self.device, self.allocator.as_mut().unwrap())?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &self.device,
            self.command_pool,
            &swapchain,
            &self.rt_pipeline_fn,
            &self.sbt,
            self.pipeline_layout,
            self.pipeline,
            self.descriptor_set,
            &self.storage_image,
        )?;

        self.swapchain = swapchain;
        self.command_buffers = command_buffers;

        Ok(())
    }

    unsafe fn cleanup_swapchain(&mut self) {
        self.device
            .free_command_buffers(self.command_pool, &self.command_buffers);
        self.command_buffers.clear();
        self.swapchain.destroy(&self.device).unwrap();
    }

    fn draw(&mut self) -> Result<bool> {
        let fence = self.fence;
        unsafe { self.device.wait_for_fences(&[fence], true, std::u64::MAX)? };

        // Drawing the frame
        let next_image_result = unsafe {
            self.swapchain.swapchain_fn.acquire_next_image(
                self.swapchain.swapchain_khr,
                std::u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match next_image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(true);
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.device.reset_fences(&[fence])? };

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.image_available_semaphore];
        let signal_semaphores = [self.render_finished_semaphore];

        let command_buffers = [self.command_buffers[image_index as usize]];
        let submit_info = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build()];
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &submit_info, fence)?
        };

        let swapchains = [self.swapchain.swapchain_khr];
        let images_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&images_indices);

        let present_result = unsafe {
            self.swapchain
                .swapchain_fn
                .queue_present(self.present_queue, &present_info)
        };
        match present_result {
            Ok(is_suboptimal) if is_suboptimal => {
                return Ok(true);
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(true);
            }
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => {}
        }
        Ok(false)
    }

    pub fn wait_for_gpu(&self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.cleanup_swapchain();
            self.device.destroy_command_pool(self.command_pool, None);
            let mut allocator = self.allocator.take().unwrap();

            self.vertex_buffer
                .destroy(&self.device, &mut allocator)
                .unwrap();
            self.index_buffer
                .destroy(&self.device, &mut allocator)
                .unwrap();

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.storage_image
                .destroy(&self.device, &mut allocator)
                .unwrap();

            self.sbt.destroy(&self.device, &mut allocator).unwrap();

            self.top_as
                .destroy(&self.device, &mut allocator, &self.acceleration_struct_fn)
                .unwrap();
            self.instance_buffer
                .destroy(&self.device, &mut allocator)
                .unwrap();
            self.bottom_as
                .destroy(&self.device, &mut allocator, &self.acceleration_struct_fn)
                .unwrap();

            drop(allocator);
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_window() -> (Window, EventLoop<()>) {
    log::debug!("Creating window and event loop");
    let events_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .with_resizable(true)
        .build(&events_loop)
        .unwrap();

    (window, events_loop)
}

fn create_vulkan_instance(
    entry: &Entry,
    window: &Window,
) -> Result<(Instance, DebugUtils, vk::DebugUtilsMessengerEXT)> {
    log::debug!("Creating vulkan instance");
    // Vulkan instance
    let app_name = CString::new(APP_NAME)?;
    let engine_name = CString::new("No Engine")?;
    let app_info = vk::ApplicationInfo::builder()
        .application_name(app_name.as_c_str())
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(engine_name.as_c_str())
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 2, 0));

    let extension_names = ash_window::enumerate_required_extensions(window)?;
    let mut extension_names = extension_names
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();
    extension_names.push(DebugUtils::name().as_ptr());

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names);

    let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

    // Vulkan debug report
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = DebugUtils::new(entry, &instance);
    let debug_utils_messenger =
        unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };

    Ok((instance, debug_utils, debug_utils_messenger))
}

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = CStr::from_ptr((*p_callback_data).p_message);
    match flag {
        Flag::VERBOSE => log::debug!("{typ:?} - {message:?}"),
        Flag::INFO => log::info!("{typ:?} - {message:?}"),
        Flag::WARNING => log::warn!("{typ:?} - {message:?}"),
        _ => log::error!("{typ:?} - {message:?}"),
    }
    vk::FALSE
}

fn create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
    instance: &Instance,
    surface: &Surface,
    surface_khr: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32)> {
    log::debug!("Creating vulkan physical device");
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut graphics = None;
    let mut present = None;
    let device = devices
        .into_iter()
        .find(|device| {
            let device = *device;

            // Does device supports graphics and present queues
            let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
            for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
                let index = index as u32;
                graphics = None;
                present = None;

                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && graphics.is_none()
                {
                    graphics = Some(index);
                }

                let present_support = unsafe {
                    surface
                        .get_physical_device_surface_support(device, index, surface_khr)
                        .expect("Failed to get device surface support")
                };
                if present_support && present.is_none() {
                    present = Some(index);
                }

                if graphics.is_some() && present.is_some() {
                    break;
                }
            }

            // Does device support desired extensions
            let required_extensions = vec![
                SwapchainFn::name(),
                RayTracingPipeline::name(),
                AccelerationStructureFn::name(),
                DeferredHostOperations::name(),
            ];
            let available_extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Failed to get device ext properties")
                    .iter()
                    .map(|ext| CStr::from_ptr(ext.extension_name.as_ptr()))
                    .collect::<Vec<_>>()
            };

            let extention_support = required_extensions
                .iter()
                .all(|e| available_extensions.contains(e));
            // Does the device have available formats for the given surface
            let formats = unsafe {
                surface
                    .get_physical_device_surface_formats(device, surface_khr)
                    .expect("Failed to get physical device surface formats")
            };

            // Does the device have available present modes for the given surface
            let present_modes = unsafe {
                surface
                    .get_physical_device_surface_present_modes(device, surface_khr)
                    .expect("Failed to get physical device surface present modes")
            };

            graphics.is_some()
                && present.is_some()
                && extention_support
                && !formats.is_empty()
                && !present_modes.is_empty()
        })
        .expect("Could not find a suitable device");

    unsafe {
        let props = instance.get_physical_device_properties(device);
        let device_name = CStr::from_ptr(props.device_name.as_ptr());
        log::debug!("Selected physical device: {device_name:?}");
    }

    Ok((device, graphics.unwrap(), present.unwrap()))
}

fn create_vulkan_device_and_graphics_and_present_qs(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
) -> Result<(Device, vk::Queue, vk::Queue)> {
    log::debug!("Creating vulkan device and graphics and present queues");
    let queue_priorities = [1.0f32];
    let queue_create_infos = {
        let mut indices = vec![graphics_q_index, present_q_index];
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

    let device_extensions_ptrs = [
        SwapchainFn::name().as_ptr(),
        RayTracingPipeline::name().as_ptr(),
        AccelerationStructureFn::name().as_ptr(),
        DeferredHostOperations::name().as_ptr(),
    ];

    let mut ray_tracing_feature =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);
    let mut acceleration_struct_feature =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder().acceleration_structure(true);
    let mut device_addr_feature =
        vk::PhysicalDeviceBufferDeviceAddressFeatures::builder().buffer_device_address(true);

    let mut features = vk::PhysicalDeviceFeatures2::builder()
        .features(vk::PhysicalDeviceFeatures::default())
        .push_next(&mut device_addr_feature)
        .push_next(&mut acceleration_struct_feature)
        .push_next(&mut ray_tracing_feature);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions_ptrs)
        .push_next(&mut features);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(graphics_q_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_q_index, 0) };

    Ok((device, graphics_queue, present_queue))
}

fn create_bottom_as(
    device: &Device,
    allocator: &mut Allocator,
    acceleration_struct_fn: &AccelerationStructureFn,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
) -> Result<(AccelerationStructure, Buffer, Buffer)> {
    // Triangle geo
    #[derive(Debug, Clone, Copy)]
    #[allow(dead_code)]
    struct Vertex {
        pos: [f32; 2],
    }

    const VERTICES: [Vertex; 3] = [
        Vertex { pos: [-1.0, 1.0] },
        Vertex { pos: [1.0, 1.0] },
        Vertex { pos: [0.0, -1.0] },
    ];

    let vertex_buffer = create_buffer(
        device,
        allocator,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::CpuToGpu,
        size_of_val(&VERTICES) as _,
        Some("Vertices"),
    )?;

    copy_data_to_buffer(&vertex_buffer, &VERTICES)?;

    const INDICES: [u16; 3] = [0, 1, 2];

    let index_buffer = create_buffer(
        device,
        allocator,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::CpuToGpu,
        size_of_val(&INDICES) as _,
        Some("Indices"),
    )?;

    copy_data_to_buffer(&index_buffer, &INDICES)?;

    let vertex_buffer_address_info =
        vk::BufferDeviceAddressInfo::builder().buffer(vertex_buffer.handle);
    let vertex_buffer_addr =
        unsafe { device.get_buffer_device_address(&vertex_buffer_address_info) };
    let index_buffer_address_info =
        vk::BufferDeviceAddressInfo::builder().buffer(index_buffer.handle);
    let index_buffer_addr = unsafe { device.get_buffer_device_address(&index_buffer_address_info) };

    let as_geo_triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_buffer_addr,
        })
        .vertex_stride(size_of::<Vertex>() as _)
        .index_type(vk::IndexType::UINT16)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_buffer_addr,
        })
        .max_vertex(INDICES.len() as _)
        .build();

    let as_struct_geo = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: as_geo_triangles_data,
        });

    let as_build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(std::slice::from_ref(&as_struct_geo));

    let as_build_size = unsafe {
        acceleration_struct_fn.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &as_build_geo_info,
            &[1],
        )
    };

    let bottom_as_buffer =
        create_as_buffer(device, allocator, as_build_size.acceleration_structure_size)?;

    let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(bottom_as_buffer.handle)
        .size(as_build_size.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let bottom_as =
        unsafe { acceleration_struct_fn.create_acceleration_structure(&as_create_info, None)? };

    let (mut scratch_buffer, scratch_buffer_addr) =
        create_scratch_buffer(device, allocator, as_build_size.build_scratch_size)?;

    let as_build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(std::slice::from_ref(&as_struct_geo))
        .dst_acceleration_structure(bottom_as)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer_addr,
        });

    let as_build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(1)
        .primitive_offset(0)
        .transform_offset(0);

    build_as(
        device,
        acceleration_struct_fn,
        command_pool,
        queue,
        &as_build_geo_info,
        &as_build_range_info,
    )?;

    let bottom_as_addr_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::builder().acceleration_structure(bottom_as);
    let bottom_as_addr = unsafe {
        acceleration_struct_fn.get_acceleration_structure_device_address(&bottom_as_addr_info)
    };

    scratch_buffer.destroy(device, allocator)?;

    let bottom_as = AccelerationStructure {
        handle: bottom_as,
        buffer: bottom_as_buffer,
        address: bottom_as_addr,
    };

    Ok((bottom_as, vertex_buffer, index_buffer))
}

fn create_top_as(
    device: &Device,
    allocator: &mut Allocator,
    acceleration_struct_fn: &AccelerationStructureFn,
    bottom_as: &AccelerationStructure,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
) -> Result<(AccelerationStructure, Buffer)> {
    #[rustfmt::skip]
    let transform_matrix = vk::TransformMatrixKHR { matrix: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]};

    let as_instance = vk::AccelerationStructureInstanceKHR {
        transform: transform_matrix,
        instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
        instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                .as_raw()
                .try_into()
                .unwrap(),
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: bottom_as.address,
        },
    };

    let instance_buffer = create_buffer(
        device,
        allocator,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::CpuToGpu,
        size_of::<vk::AccelerationStructureInstanceKHR>() as _,
        Some("Instance"),
    )?;

    copy_data_to_buffer(&instance_buffer, &[as_instance])?;

    let instance_buffer_addr_info =
        vk::BufferDeviceAddressInfo::builder().buffer(instance_buffer.handle);
    let instance_buffer_addr =
        unsafe { device.get_buffer_device_address(&instance_buffer_addr_info) };

    let as_geo = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer_addr,
                })
                .build(),
        });

    let as_build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(std::slice::from_ref(&as_geo));

    let as_build_size = unsafe {
        acceleration_struct_fn.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &as_build_geo_info,
            &[1],
        )
    };

    let top_as_buffer =
        create_as_buffer(device, allocator, as_build_size.acceleration_structure_size)?;

    let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(top_as_buffer.handle)
        .size(as_build_size.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
    let top_as =
        unsafe { acceleration_struct_fn.create_acceleration_structure(&as_create_info, None)? };

    let (mut scratch_buffer, scratch_buffer_addr) =
        create_scratch_buffer(device, allocator, as_build_size.build_scratch_size)?;

    let as_build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(std::slice::from_ref(&as_geo))
        .dst_acceleration_structure(top_as)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer_addr,
        });

    let as_build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(1)
        .primitive_offset(0)
        .transform_offset(0);

    build_as(
        device,
        acceleration_struct_fn,
        command_pool,
        queue,
        &as_build_geo_info,
        &as_build_range_info,
    )?;

    let top_as_addr_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::builder().acceleration_structure(top_as);
    let top_as_addr = unsafe {
        acceleration_struct_fn.get_acceleration_structure_device_address(&top_as_addr_info)
    };

    scratch_buffer.destroy(device, allocator)?;

    let top_as = AccelerationStructure {
        handle: top_as,
        buffer: top_as_buffer,
        address: top_as_addr,
    };

    Ok((top_as, instance_buffer))
}

fn create_storage_image(
    device: &Device,
    allocator: &mut Allocator,
    format: vk::Format,
    extent: vk::Extent2D,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
) -> Result<ImageAndView> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&image_info, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator.allocate(&AllocationCreateDesc {
        name: "Image storage",
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: true,
    })?;

    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

    let view_info = vk::ImageViewCreateInfo::builder()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image(image);

    let view = unsafe { device.create_image_view(&view_info, None)? };

    execute_one_time_commands(device, queue, command_pool, |cmd_buffer| {
        transition_image_layout(
            device,
            cmd_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
        );
    })?;

    Ok(ImageAndView {
        image,
        view,
        allocation: Some(allocation),
    })
}

fn create_pipeline(
    device: &Device,
    rt_pipeline_fn: &RayTracingPipeline,
) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)> {
    // descriptor and pipeline layouts
    let as_layout_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
    ];

    let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&as_layout_bindings);
    let dsl = unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

    let pipe_layout_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&dsl));
    let pipe_layout = unsafe { device.create_pipeline_layout(&pipe_layout_info, None)? };

    // shader groups
    let raygen_source = read_shader_from_bytes(
        &include_bytes!("../../assets/shaders/triangle_basic/raygen.rgen.spv")[..],
    )?;
    let raygen_create_info = vk::ShaderModuleCreateInfo::builder().code(&raygen_source);
    let raygen_module = unsafe { device.create_shader_module(&raygen_create_info, None)? };

    let miss_source = read_shader_from_bytes(
        &include_bytes!("../../assets/shaders/triangle_basic/miss.rmiss.spv")[..],
    )?;
    let miss_create_info = vk::ShaderModuleCreateInfo::builder().code(&miss_source);
    let miss_module = unsafe { device.create_shader_module(&miss_create_info, None)? };

    let closesthit_source = read_shader_from_bytes(
        &include_bytes!("../../assets/shaders/triangle_basic/closesthit.rchit.spv")[..],
    )?;
    let closesthit_create_info = vk::ShaderModuleCreateInfo::builder().code(&closesthit_source);
    let closesthit_module = unsafe { device.create_shader_module(&closesthit_create_info, None)? };

    let entry_point_name = CString::new("main")?;

    let shader_stages_infos = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(raygen_module)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(miss_module)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(closesthit_module)
            .name(&entry_point_name)
            .build(),
    ];

    let shader_groups_infos = [
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(2)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
    ];

    let pipe_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .stages(&shader_stages_infos)
        .groups(&shader_groups_infos)
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipe_layout);
    let pipe = unsafe {
        rt_pipeline_fn.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipe_info),
            None,
        )?
    };

    unsafe {
        device.destroy_shader_module(raygen_module, None);
        device.destroy_shader_module(miss_module, None);
        device.destroy_shader_module(closesthit_module, None);
    }

    Ok((dsl, pipe_layout, pipe[0]))
}

fn create_shader_binding_table(
    device: &Device,
    allocator: &mut Allocator,
    rt_pipeline_fn: &RayTracingPipeline,
    rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pipeline: vk::Pipeline,
) -> Result<Sbt> {
    let handle_size = rt_pipeline_properties.shader_group_handle_size;
    let handle_alignment = rt_pipeline_properties.shader_group_handle_alignment;
    let base_alignment = rt_pipeline_properties.shader_group_base_alignment;

    let aligned_handle_size = compute_aligned_size(handle_size, handle_alignment);
    let aligned_base_size = compute_aligned_size(aligned_handle_size, base_alignment);

    let group_count = 3;
    let data_size = group_count * handle_size;

    let handles = unsafe {
        rt_pipeline_fn.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            group_count,
            data_size as _,
        )?
    };

    let buffer_usage = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let memory_location = MemoryLocation::CpuToGpu;

    let raygen_region_size = aligned_base_size;
    let miss_region_size = aligned_base_size;
    let hit_region_size = aligned_base_size;
    let buffer_size = raygen_region_size + miss_region_size + hit_region_size;

    let sbt_buffer = create_buffer(
        device,
        allocator,
        buffer_usage,
        memory_location,
        buffer_size as _,
        Some("sbt"),
    )?;

    // This just works because the STB contains 3 groups of 1 handle
    let mut stb_data = Vec::<u8>::with_capacity(buffer_size as _);
    for i in 0..group_count as usize {
        let offset = i * handle_size as usize;
        let pad = (aligned_base_size - handle_size) as usize;
        for j in 0..handle_size as usize {
            stb_data.push(handles[offset + j]);
        }
        for _k in 0..pad {
            stb_data.push(0);
        }
    }

    copy_data_to_buffer(&sbt_buffer, &stb_data)?;

    let stb_addr_info = vk::BufferDeviceAddressInfo::builder().buffer(sbt_buffer.handle);
    let stb_addr = unsafe { device.get_buffer_device_address(&stb_addr_info) };

    // see https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/Images/sbt_0.png
    let raygen_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(stb_addr)
        .size(raygen_region_size as _)
        .stride(raygen_region_size as _)
        .build();

    let miss_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(stb_addr + raygen_region.size)
        .size(miss_region_size as _)
        .stride(aligned_handle_size as _)
        .build();

    let hit_region = vk::StridedDeviceAddressRegionKHR::builder()
        .device_address(stb_addr + raygen_region.size + miss_region.size)
        .size(hit_region_size as _)
        .stride(aligned_handle_size as _)
        .build();

    let sbt = Sbt {
        buffer: sbt_buffer,
        _address: stb_addr,
        raygen_region,
        miss_region,
        hit_region,
    };

    Ok(sbt)
}
fn create_descriptor_set(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    top_as: &AccelerationStructure,
    storage_img: &ImageAndView,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .build(),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

    let sets_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));
    let set = unsafe { device.allocate_descriptor_sets(&sets_alloc_info)?[0] };

    let mut write_set_as = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
        .acceleration_structures(std::slice::from_ref(&top_as.handle));

    let mut as_write_set = vk::WriteDescriptorSet::builder()
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .dst_set(set)
        .dst_binding(0)
        .push_next(&mut write_set_as)
        .build();
    as_write_set.descriptor_count = 1;

    let desc_img_info = vk::DescriptorImageInfo::builder()
        .image_view(storage_img.view)
        .image_layout(vk::ImageLayout::GENERAL);

    let img_write_set = vk::WriteDescriptorSet::builder()
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_set(set)
        .dst_binding(1)
        .image_info(std::slice::from_ref(&desc_img_info))
        .build();

    let write_sets = [as_write_set, img_write_set];
    unsafe { device.update_descriptor_sets(&write_sets, &[]) };

    Ok((pool, set))
}

fn create_as_buffer(
    device: &Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
) -> Result<Buffer> {
    create_buffer(
        device,
        allocator,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        size,
        Some("AS"),
    )
}

fn create_scratch_buffer(
    device: &Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
) -> Result<(Buffer, u64)> {
    let buffer = create_buffer(
        device,
        allocator,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        size,
        Some("Scratch"),
    )?;

    let addr_info = vk::BufferDeviceAddressInfo::builder().buffer(buffer.handle);
    let addr = unsafe { device.get_buffer_device_address(&addr_info) };

    Ok((buffer, addr))
}

fn create_buffer(
    device: &Device,
    allocator: &mut Allocator,
    usage: vk::BufferUsageFlags,
    memory_location: MemoryLocation,
    size: vk::DeviceSize,
    debug_allocation_name: Option<&str>,
) -> Result<Buffer> {
    let create_info = vk::BufferCreateInfo::builder().size(size).usage(usage);
    let buffer = unsafe { device.create_buffer(&create_info, None)? };
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name: debug_allocation_name.unwrap_or(""),
        requirements,
        location: memory_location,
        linear: true,
    })?;

    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };

    Ok(Buffer {
        handle: buffer,
        allocation: Some(allocation),
    })
}

fn copy_data_to_buffer<T: Copy>(buffer: &Buffer, data: &[T]) -> Result<()> {
    unsafe {
        let data_ptr = buffer
            .allocation
            .as_ref()
            .unwrap()
            .mapped_ptr()
            .unwrap()
            .as_ptr();
        let mut align =
            ash::util::Align::new(data_ptr, align_of::<T>() as _, size_of_val(data) as _);
        align.copy_from_slice(data);
    };

    Ok(())
}

fn build_as(
    device: &Device,
    acceleration_struct_fn: &AccelerationStructureFn,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    as_build_geo_info: &vk::AccelerationStructureBuildGeometryInfoKHR,
    as_build_range_info: &vk::AccelerationStructureBuildRangeInfoKHR,
) -> Result<()> {
    let result = execute_one_time_commands(device, queue, command_pool, |cmd_buffer| {
        unsafe {
            acceleration_struct_fn.cmd_build_acceleration_structures(
                cmd_buffer,
                std::slice::from_ref(as_build_geo_info),
                std::slice::from_ref(&std::slice::from_ref(as_build_range_info)),
            )
        };
    })?;

    Ok(result)
}

fn execute_one_time_commands<R, F: FnOnce(vk::CommandBuffer) -> R>(
    device: &Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    executor: F,
) -> Result<R> {
    let command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool)
            .command_buffer_count(1);

        unsafe { device.allocate_command_buffers(&alloc_info)?[0] }
    };
    let command_buffers = [command_buffer];

    // Begin recording
    {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };
    }

    // Execute user function
    let executor_result = executor(command_buffer);

    // End recording
    unsafe { device.end_command_buffer(command_buffer)? };

    // Submit and wait
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .build();

    let fence_info = vk::FenceCreateInfo::builder();
    let fence = unsafe { device.create_fence(&fence_info, None)? };

    unsafe {
        device.queue_submit(queue, std::slice::from_ref(&submit_info), fence)?;
        device.wait_for_fences(&[fence], true, std::u64::MAX)?;
    };

    // Free
    unsafe {
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, &command_buffers)
    };

    Ok(executor_result)
}

fn transition_image_layout(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .build();

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };
}

fn compute_aligned_size(size: u32, alignment: u32) -> u32 {
    (size + (alignment - 1)) & !(alignment - 1)
}

fn create_vulkan_swapchain(
    instance: &Instance,
    surface: &Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
    device: &Device,
) -> Result<Swapchain> {
    log::debug!("Creating vulkan swapchain");
    // Swapchain format
    let format = {
        let formats =
            unsafe { surface.get_physical_device_surface_formats(physical_device, surface_khr)? };
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
            surface
                .get_physical_device_surface_present_modes(physical_device, surface_khr)
                .expect("Failed to get physical device surface present modes")
        };
        if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    };
    log::debug!("Swapchain present mode: {present_mode:?}");

    let capabilities =
        unsafe { surface.get_physical_device_surface_capabilities(physical_device, surface_khr)? };

    // Swapchain extent
    let extent = {
        if capabilities.current_extent.width != std::u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = WIDTH.min(max.width).max(min.width);
            let height = HEIGHT.min(max.height).max(min.height);
            vk::Extent2D { width, height }
        }
    };
    log::debug!("Swapchain extent: {extent:?}");

    // Swapchain image count
    let image_count = capabilities.min_image_count;
    log::debug!("Swapchain image count: {image_count:?}");

    // Swapchain
    let families_indices = [graphics_q_index, present_q_index];
    let create_info = {
        let mut builder = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface_khr)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST);

        builder = if graphics_q_index != present_q_index {
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

    let swapchain_fn = SwapchainFn::new(instance, device);
    let swapchain_khr = unsafe { swapchain_fn.create_swapchain(&create_info, None)? };

    // Swapchain images and image views
    let images = unsafe { swapchain_fn.get_swapchain_images(swapchain_khr)? };
    let views = images
        .iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe { device.create_image_view(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Swapchain {
        swapchain_fn,
        swapchain_khr,
        extent,
        format: format.format,
        images,
        views,
    })
}

fn read_shader_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut cursor = std::io::Cursor::new(bytes);
    Ok(ash::util::read_spv(&mut cursor)?)
}

fn create_and_record_command_buffers(
    device: &Device,
    pool: vk::CommandPool,
    swapchain: &Swapchain,
    rt_pipeline_fn: &RayTracingPipeline,
    sbt: &Sbt,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
    storage_image: &ImageAndView,
) -> Result<Vec<vk::CommandBuffer>> {
    log::debug!("Creating and recording command buffers");
    let buffers = {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain.images.len() as _);

        unsafe { device.allocate_command_buffers(&allocate_info)? }
    };

    for (index, buffer) in buffers.iter().enumerate() {
        let buffer = *buffer;
        let swapchain_image = swapchain.images[index];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
        unsafe { device.begin_command_buffer(buffer, &command_buffer_begin_info)? };

        unsafe {
            device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline)
        };

        let descriptor_set = [descriptor_set];
        unsafe {
            device.cmd_bind_descriptor_sets(
                buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline_layout,
                0,
                &descriptor_set,
                &[],
            )
        }

        let extent = swapchain.extent;

        unsafe {
            let empty_region = vk::StridedDeviceAddressRegionKHR::builder();
            rt_pipeline_fn.cmd_trace_rays(
                buffer,
                &sbt.raygen_region,
                &sbt.miss_region,
                &sbt.hit_region,
                &empty_region,
                extent.width,
                extent.height,
                1,
            );
        }

        transition_image_layout(
            device,
            buffer,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        transition_image_layout(
            device,
            buffer,
            storage_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        let copy_region = vk::ImageCopy::builder()
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
                width: extent.width,
                height: extent.height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_image(
                buffer,
                storage_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            )
        };

        transition_image_layout(
            device,
            buffer,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        transition_image_layout(
            device,
            buffer,
            storage_image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TOP_OF_PIPE,
        );

        unsafe { device.end_command_buffer(buffer)? };
    }

    Ok(buffers)
}
