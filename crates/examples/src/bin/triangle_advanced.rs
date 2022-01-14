use anyhow::Result;
use ash::vk::{self, Packed24_8};
use gpu_allocator::MemoryLocation;
use simple_logger::SimpleLogger;
use std::mem::size_of;
use vulkan::utils::*;
use vulkan::*;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 576;
const IN_FLIGHT_FRAMES: u32 = 2;
const APP_NAME: &str = "Triangle advanced";

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
                        app.recreate_swapchain(dim.width, dim.height)
                            .expect("Failed to recreate swapchain");
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

struct BottomAS {
    inner: VkAccelerationStructure,
    _vertex_buffer: VkBuffer,
    _index_buffer: VkBuffer,
}

struct TopAS {
    inner: VkAccelerationStructure,
    _instance_buffer: VkBuffer,
}

struct ImageAndView {
    view: VkImageView,
    image: VkImage,
}

struct PipelineRes {
    pipeline: VkRTPipeline,
    pipeline_layout: VkPipelineLayout,
    static_dsl: VkDescriptorSetLayout,
    dynamic_dsl: VkDescriptorSetLayout,
}

struct DescriptorRes {
    _pool: VkDescriptorPool,
    static_set: VkDescriptorSet,
    dynamic_sets: Vec<VkDescriptorSet>,
}

struct App {
    swapchain: VkSwapchain,
    command_pool: VkCommandPool,
    pipeline_res: PipelineRes,
    _bottom_as: BottomAS,
    _top_as: TopAS,
    storage_images: Vec<ImageAndView>,
    descriptor_res: DescriptorRes,
    sbt: VkShaderBindingTable,
    command_buffers: Vec<VkCommandBuffer>,
    in_flight_frames: InFlightFrames,
    context: VkContext,
}

impl App {
    fn new(window: &Window) -> Result<Self> {
        log::info!("Create application");

        // Vulkan context
        let required_extensions = [
            "VK_KHR_swapchain",
            "VK_KHR_ray_tracing_pipeline",
            "VK_KHR_acceleration_structure",
            "VK_KHR_deferred_host_operations",
        ];
        let mut context = VkContext::new(
            window,
            VkVersion::from_major_minor(1, 2),
            Some(APP_NAME),
            &required_extensions,
        )?;

        // Command pool
        let command_pool = context.create_command_pool(context.graphics_queue_family, None)?;

        // Swapchain
        let swapchain = VkSwapchain::new(&context, WIDTH, HEIGHT)?;

        // Bottom AS
        let bottom_as = create_bottom_as(&mut context)?;

        // Top AS
        let top_as = create_top_as(&mut context, &bottom_as)?;

        // Storage image
        let storage_images = create_storage_images(
            &mut context,
            swapchain.format,
            swapchain.extent,
            swapchain.images.len(),
        )?;

        // RT pipeline
        let pipeline_res = create_pipeline(&context)?;

        // Shader Binding Table (SBT)
        let sbt = context.create_shader_binding_table(&pipeline_res.pipeline)?;

        // RT Descriptor sets
        let descriptor_res =
            create_descriptor_sets(&context, &pipeline_res, &top_as, &storage_images)?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &command_pool,
            &swapchain,
            &sbt,
            &pipeline_res,
            &descriptor_res,
            &storage_images,
        )?;

        // Semaphore use for presentation
        let in_flight_frames = InFlightFrames::new(&context, IN_FLIGHT_FRAMES)?;

        Ok(Self {
            context,
            command_pool,
            swapchain,
            pipeline_res,
            _bottom_as: bottom_as,
            _top_as: top_as,
            storage_images,
            descriptor_res,
            sbt,
            command_buffers,
            in_flight_frames,
        })
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        log::debug!("Recreating the swapchain");

        self.wait_for_gpu()?;

        // Swapchain
        unsafe { self.cleanup_swapchain_dependent_resources() };
        self.swapchain.resize(&self.context, width, height)?;

        // Recreate storage image for RT and update descriptor set
        let storage_images = create_storage_images(
            &mut self.context,
            self.swapchain.format,
            self.swapchain.extent,
            self.swapchain.images.len(),
        )?;

        storage_images.iter().enumerate().for_each(|(index, img)| {
            let set = &self.descriptor_res.dynamic_sets[index];
            let img_write_set = VkWriteDescriptorSet {
                binding: 1,
                kind: VkWriteDescriptorSetKind::StorageImage {
                    layout: vk::ImageLayout::GENERAL,
                    view: &img.view,
                },
            };

            set.update(&img_write_set);
        });

        let _ = std::mem::replace(&mut self.storage_images, storage_images);

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &self.command_pool,
            &self.swapchain,
            &self.sbt,
            &self.pipeline_res,
            &self.descriptor_res,
            &self.storage_images,
        )?;

        self.command_buffers = command_buffers;

        Ok(())
    }

    unsafe fn cleanup_swapchain_dependent_resources(&mut self) {
        self.command_pool
            .free_command_buffers(&self.command_buffers);
        self.command_buffers.clear();
    }

    fn draw(&mut self) -> Result<bool> {
        let SyncObjects {
            image_available_semaphore,
            render_finished_semaphore,
            fence,
        } = self.in_flight_frames.next();

        fence.wait(None)?;

        // Drawing the frame
        let next_image_result = self
            .swapchain
            .acquire_next_image(std::u64::MAX, image_available_semaphore);
        let image_index = match next_image_result {
            Ok(AcquiredImage { index, .. }) => index,
            Err(err) => match err.downcast_ref::<vk::Result>() {
                Some(&vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                _ => panic!("Error while acquiring next image. Cause: {}", err),
            },
        };

        fence.reset()?;

        let command_buffer = &self.command_buffers[image_index as usize];
        self.context.graphics_queue.submit(
            command_buffer,
            Some(image_available_semaphore),
            Some(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
            Some(render_finished_semaphore),
            fence,
        )?;

        let signal_semaphores = [render_finished_semaphore];
        let present_result = self.swapchain.queue_present(
            image_index,
            &signal_semaphores,
            &self.context.present_queue,
        );
        match present_result {
            Ok(true) => return Ok(true),
            Err(err) => match err.downcast_ref::<vk::Result>() {
                Some(&vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                _ => panic!("Failed to present queue. Cause: {}", err),
            },
            _ => {}
        }

        Ok(false)
    }

    pub fn wait_for_gpu(&self) -> Result<()> {
        self.context.device_wait_idle()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.cleanup_swapchain_dependent_resources();
        }
    }
}

fn create_bottom_as(context: &mut VkContext) -> Result<BottomAS> {
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

    let vertex_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &VERTICES,
    )?;
    let vertex_buffer_addr = vertex_buffer.get_device_address();

    const INDICES: [u16; 3] = [0, 1, 2];

    let index_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &INDICES,
    )?;
    let index_buffer_addr = index_buffer.get_device_address();

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
        })
        .build();

    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(1)
        .primitive_offset(0)
        .transform_offset(0)
        .build();

    let inner = context.create_bottom_level_acceleration_structure(
        &[as_struct_geo],
        &[build_range_info],
        &[1],
    )?;

    Ok(BottomAS {
        inner,
        _vertex_buffer: vertex_buffer,
        _index_buffer: index_buffer,
    })
}

fn create_top_as(context: &mut VkContext, bottom_as: &BottomAS) -> Result<TopAS> {
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
            device_handle: bottom_as.inner.address,
        },
    };

    let instance_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &[as_instance],
    )?;
    let instance_buffer_addr = instance_buffer.get_device_address();

    let as_struct_geo = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer_addr,
                })
                .build(),
        })
        .build();

    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(1)
        .primitive_offset(0)
        .transform_offset(0)
        .build();

    let inner = context.create_top_level_acceleration_structure(
        &[as_struct_geo],
        &[build_range_info],
        &[1],
    )?;

    Ok(TopAS {
        inner,
        _instance_buffer: instance_buffer,
    })
}

fn create_storage_images(
    context: &mut VkContext,
    format: vk::Format,
    extent: vk::Extent2D,
    count: usize,
) -> Result<Vec<ImageAndView>> {
    let mut images = Vec::with_capacity(count);

    for _ in 0..count {
        let image = context.create_image(
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
            MemoryLocation::GpuOnly,
            format,
            extent.width,
            extent.height,
        )?;

        let view = image.create_image_view()?;

        context.execute_one_time_commands(|cmd_buffer| {
            cmd_buffer.transition_layout(
                &image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
            );
        })?;

        images.push(ImageAndView { image, view })
    }

    Ok(images)
}

fn create_pipeline(context: &VkContext) -> Result<PipelineRes> {
    // descriptor and pipeline layouts
    let static_layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
        .build()];

    let dynamic_layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
        .build()];

    let static_dsl = context.create_descriptor_set_layout(&static_layout_bindings)?;
    let dynamic_dsl = context.create_descriptor_set_layout(&dynamic_layout_bindings)?;
    let dsls = [&static_dsl, &dynamic_dsl];

    let pipeline_layout = context.create_pipeline_layout(&dsls)?;

    // Shaders
    let shaders_create_info = [
        VkRTShaderCreateInfo {
            source: &include_bytes!("../../../../assets/shaders/triangle_advanced/raygen.rgen.spv")
                [..],
            stage: vk::ShaderStageFlags::RAYGEN_KHR,
            group: VkRTShaderGroup::RayGen,
        },
        VkRTShaderCreateInfo {
            source: &include_bytes!("../../../../assets/shaders/triangle_advanced/miss.rmiss.spv")
                [..],
            stage: vk::ShaderStageFlags::MISS_KHR,
            group: VkRTShaderGroup::Miss,
        },
        VkRTShaderCreateInfo {
            source: &include_bytes!(
                "../../../../assets/shaders/triangle_advanced/closesthit.rchit.spv"
            )[..],
            stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            group: VkRTShaderGroup::ClosestHit,
        },
    ];

    let pipeline_create_info = VkRTPipelineCreateInfo {
        shaders: &shaders_create_info,
        max_ray_recursion_depth: 1,
    };

    let pipeline = context.create_ray_tracing_pipeline(&pipeline_layout, &pipeline_create_info)?;

    Ok(PipelineRes {
        pipeline,
        pipeline_layout,
        static_dsl,
        dynamic_dsl,
    })
}

fn create_descriptor_sets(
    context: &VkContext,
    pipeline_res: &PipelineRes,
    top_as: &TopAS,
    storage_imgs: &[ImageAndView],
) -> Result<DescriptorRes> {
    let set_count = storage_imgs.len() as u32;

    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(set_count)
            .build(),
    ];

    let pool = context.create_descriptor_pool(set_count + 1, &pool_sizes)?;

    let static_set = pool.allocate_set(&pipeline_res.static_dsl)?;
    let dynamic_sets = pool.allocate_sets(&pipeline_res.dynamic_dsl, set_count)?;

    static_set.update(&VkWriteDescriptorSet {
        binding: 0,
        kind: VkWriteDescriptorSetKind::AccelerationStructure {
            acceleration_structure: &top_as.inner,
        },
    });

    dynamic_sets.iter().enumerate().for_each(|(index, set)| {
        set.update(&VkWriteDescriptorSet {
            binding: 1,
            kind: VkWriteDescriptorSetKind::StorageImage {
                layout: vk::ImageLayout::GENERAL,
                view: &storage_imgs[index].view,
            },
        });
    });

    Ok(DescriptorRes {
        _pool: pool,
        dynamic_sets,
        static_set,
    })
}

fn create_and_record_command_buffers(
    pool: &VkCommandPool,
    swapchain: &VkSwapchain,
    sbt: &VkShaderBindingTable,
    pipeline_res: &PipelineRes,
    descriptor_res: &DescriptorRes,
    storage_images: &[ImageAndView],
) -> Result<Vec<VkCommandBuffer>> {
    log::debug!("Creating and recording command buffers");
    let buffers = pool
        .allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, swapchain.images.len() as _)?;

    let static_set = &descriptor_res.static_set;

    for (index, buffer) in buffers.iter().enumerate() {
        let dynamic_set = &descriptor_res.dynamic_sets[index];
        let swapchain_image = &swapchain.images[index];
        let storage_image = &storage_images[index].image;

        buffer.begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;

        buffer.bind_pipeline(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            &pipeline_res.pipeline,
        );

        buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            &pipeline_res.pipeline_layout,
            0,
            &[static_set, dynamic_set],
        );

        buffer.trace_rays(sbt, swapchain.extent.width, swapchain.extent.height);

        buffer.transition_layout(
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        buffer.transition_layout(
            storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        buffer.copy_image(
            storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        buffer.transition_layout(
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        buffer.transition_layout(
            storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TOP_OF_PIPE,
        );

        buffer.end()?;
    }

    Ok(buffers)
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(context: &VkContext, frame_count: u32) -> Result<Self> {
        let sync_objects = (0..frame_count)
            .map(|_i| {
                let image_available_semaphore = context.create_semaphore()?;
                let render_finished_semaphore = context.create_semaphore()?;
                let fence = context.create_fence(Some(vk::FenceCreateFlags::SIGNALED))?;

                Ok(SyncObjects {
                    image_available_semaphore,
                    render_finished_semaphore,
                    fence,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            sync_objects,
            current_frame: 0,
        })
    }

    fn next(&mut self) -> &SyncObjects {
        let next = &self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        next
    }
}

struct SyncObjects {
    image_available_semaphore: VkSemaphore,
    render_finished_semaphore: VkSemaphore,
    fence: VkFence,
}
