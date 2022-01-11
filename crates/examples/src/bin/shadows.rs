use anyhow::Result;
use ash::vk::{self, Packed24_8};
use glam::{vec3, Mat4};
use gltf::Vertex;
use gpu_allocator::MemoryLocation;
use simple_logger::SimpleLogger;
use std::{
    ffi::CString,
    mem::{size_of, size_of_val},
};
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
const APP_NAME: &str = "Shadows";

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
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    geometry_info_buffer: VkBuffer,
}

struct TopAS {
    inner: VkAccelerationStructure,
    _instance_buffer: VkBuffer,
}

struct ImageAndView {
    view: VkImageView,
    image: VkImage,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CameraUBO {
    inverted_view: Mat4,
    inverted_proj: Mat4,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GeometryInfo {
    transform: Mat4,
    vertex_offset: u32,
    index_offset: u32,
}

struct App {
    swapchain: VkSwapchain,
    command_pool: VkCommandPool,
    _descriptor_set_layout: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
    _ubo_buffer: VkBuffer,
    _bottom_as: BottomAS,
    _top_as: TopAS,
    storage_images: Vec<ImageAndView>,
    _descriptor_pool: VkDescriptorPool,
    descriptor_sets: VkDescriptorSets,
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

        // UBO
        let ubo_buffer = create_ubo_buffer(&context)?;

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
        let (descriptor_set_layout, pipeline_layout, pipeline) = create_pipeline(&context)?;

        // Shader Binding Table (SBT)
        let sbt = context.create_shader_binding_table(
            &pipeline,
            VkShaderBindingTableDesc {
                group_count: 4,
                raygen_shader_count: 1,
                miss_shader_count: 2,
                hit_shader_count: 1,
            },
        )?;

        // RT Descriptor sets
        let (descriptor_pool, descriptor_sets) = create_descriptor_sets(
            &context,
            &descriptor_set_layout,
            &bottom_as,
            &top_as,
            &storage_images,
            &ubo_buffer,
        )?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &command_pool,
            &swapchain,
            &sbt,
            &pipeline_layout,
            &pipeline,
            &descriptor_sets,
            &storage_images,
        )?;

        // Semaphore use for presentation
        let in_flight_frames = InFlightFrames::new(&context, IN_FLIGHT_FRAMES)?;

        Ok(Self {
            context,
            command_pool,
            swapchain,
            _descriptor_set_layout: descriptor_set_layout,
            pipeline_layout,
            pipeline,
            _ubo_buffer: ubo_buffer,
            _bottom_as: bottom_as,
            _top_as: top_as,
            storage_images,
            _descriptor_pool: descriptor_pool,
            descriptor_sets,
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
            let set = &self.descriptor_sets.sets[index];
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
            &self.pipeline_layout,
            &self.pipeline,
            &self.descriptor_sets,
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

fn create_ubo_buffer(context: &VkContext) -> Result<VkBuffer> {
    let view = Mat4::look_at_rh(
        vec3(-1.0, 1.5, 3.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    let inverted_view = view.inverse();

    let proj = Mat4::perspective_infinite_rh(60f32.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1);
    let inverted_proj = proj.inverse();

    let cam_ubo = CameraUBO {
        inverted_view,
        inverted_proj,
    };

    let ubo_buffer = context.create_buffer(
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        MemoryLocation::CpuToGpu,
        size_of_val(&cam_ubo) as _,
    )?;

    ubo_buffer.copy_data_to_buffer(&[cam_ubo])?;

    Ok(ubo_buffer)
}

fn create_bottom_as(context: &mut VkContext) -> Result<BottomAS> {
    let model = gltf::load_file("./crates/examples/assets/models/cesium_man_with_light.glb")?;
    let vertices = model.vertices.as_slice();
    let indices = model.indices.as_slice();

    let vertex_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        vertices,
    )?;
    let vertex_buffer_addr = vertex_buffer.get_device_address();

    let index_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        indices,
    )?;
    let index_buffer_addr = index_buffer.get_device_address();

    let transforms = model
        .nodes
        .iter()
        .map(|n| {
            let transform = n.transform;
            let r0 = transform[0];
            let r1 = transform[1];
            let r2 = transform[2];
            let r3 = transform[3];

            #[rustfmt::skip]
            let matrix = [
                r0[0], r1[0], r2[0], r3[0],
                r0[1], r1[1], r2[1], r3[1],
                r0[2], r1[2], r2[2], r3[2],
            ];

            vk::TransformMatrixKHR { matrix }
        })
        .collect::<Vec<_>>();
    let transform_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &transforms,
    )?;
    let transform_buffer_addr = transform_buffer.get_device_address();

    let as_geo_triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_buffer_addr,
        })
        .vertex_stride(size_of::<Vertex>() as _)
        .max_vertex(vertices.len() as _)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_buffer_addr,
        })
        .transform_data(vk::DeviceOrHostAddressConstKHR {
            device_address: transform_buffer_addr,
        })
        .build();

    let mut geometry_infos = vec![];
    let mut as_geometries = vec![];
    let mut as_ranges = vec![];
    let mut max_primitive_counts = vec![];

    for (node_index, node) in model.nodes.iter().enumerate() {
        let mesh = node.mesh;

        let primitive_count = (mesh.index_count / 3) as u32;

        geometry_infos.push(GeometryInfo {
            transform: Mat4::from_cols_array_2d(&node.transform),
            vertex_offset: mesh.vertex_offset,
            index_offset: mesh.index_offset,
        });

        as_geometries.push(
            vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: as_geo_triangles_data,
                })
                .build(),
        );

        as_ranges.push(
            vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .first_vertex(mesh.vertex_offset)
                .primitive_count(primitive_count)
                .primitive_offset(mesh.index_offset * size_of::<u32>() as u32)
                .transform_offset((node_index * size_of::<vk::TransformMatrixKHR>()) as u32)
                .build(),
        );

        max_primitive_counts.push(primitive_count)
    }

    let geometry_info_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &geometry_infos,
    )?;

    let inner = context.create_bottom_level_acceleration_structure(
        &as_geometries,
        &as_ranges,
        &max_primitive_counts,
    )?;

    Ok(BottomAS {
        inner,
        vertex_buffer,
        index_buffer,
        geometry_info_buffer,
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
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
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

    let as_ranges = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .first_vertex(0)
        .primitive_count(1)
        .primitive_offset(0)
        .transform_offset(0)
        .build();

    let inner =
        context.create_top_level_acceleration_structure(&[as_struct_geo], &[as_ranges], &[1])?;

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

fn create_pipeline(
    context: &VkContext,
) -> Result<(VkDescriptorSetLayout, VkPipelineLayout, VkPipeline)> {
    // descriptor and pipeline layouts
    let as_layout_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
        // Vertex buffer
        vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        //Index buffer
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        // Geometry info buffer
        vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
    ];

    let dsl = context.create_descriptor_set_layout(&as_layout_bindings)?;

    let pipe_layout = context.create_pipeline_layout(&dsl)?;

    // shader groups
    let raygen_module = context.create_shader_module(
        &include_bytes!("../../assets/shaders/shadows/raygen.rgen.spv")[..],
    )?;
    let miss_module = context
        .create_shader_module(&include_bytes!("../../assets/shaders/shadows/miss.rmiss.spv")[..])?;
    let shadow_miss = context.create_shader_module(
        &include_bytes!("../../assets/shaders/shadows/shadow.rmiss.spv")[..],
    )?;
    let closesthit_module = context.create_shader_module(
        &include_bytes!("../../assets/shaders/shadows/closesthit.rchit.spv")[..],
    )?;

    let entry_point_name = CString::new("main").unwrap();

    let shader_stages_infos = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(raygen_module.inner)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(miss_module.inner)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(shadow_miss.inner)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(closesthit_module.inner)
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
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(2)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(3)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build(),
    ];

    // TODO: abstract
    let mut pipe_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .stages(&shader_stages_infos)
        .groups(&shader_groups_infos)
        .max_pipeline_ray_recursion_depth(2);

    let pipe = context.create_ray_tracing_pipeline(&pipe_layout, &mut pipe_info)?;

    Ok((dsl, pipe_layout, pipe))
}

fn create_descriptor_sets(
    context: &VkContext,
    descriptor_set_layout: &VkDescriptorSetLayout,
    bottom_as: &BottomAS,
    top_as: &TopAS,
    storage_imgs: &[ImageAndView],
    ubo_buffer: &VkBuffer,
) -> Result<(VkDescriptorPool, VkDescriptorSets)> {
    let set_count = storage_imgs.len() as u32;

    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(set_count as _)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(set_count)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(set_count)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(set_count * 3)
            .build(),
    ];

    let pool = context.create_descriptor_pool(set_count as _, &pool_sizes)?;

    let sets = pool.allocate_sets(descriptor_set_layout, set_count)?;

    sets.iter().enumerate().for_each(|(index, set)| {
        set.update(&VkWriteDescriptorSet {
            binding: 0,
            kind: VkWriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: &top_as.inner,
            },
        });

        set.update(&VkWriteDescriptorSet {
            binding: 1,
            kind: VkWriteDescriptorSetKind::StorageImage {
                layout: vk::ImageLayout::GENERAL,
                view: &storage_imgs[index].view,
            },
        });

        set.update(&VkWriteDescriptorSet {
            binding: 2,
            kind: VkWriteDescriptorSetKind::UniformBuffer { buffer: ubo_buffer },
        });

        set.update(&VkWriteDescriptorSet {
            binding: 3,
            kind: VkWriteDescriptorSetKind::StorageBuffer {
                buffer: &bottom_as.vertex_buffer,
            },
        });

        set.update(&VkWriteDescriptorSet {
            binding: 4,
            kind: VkWriteDescriptorSetKind::StorageBuffer {
                buffer: &bottom_as.index_buffer,
            },
        });

        set.update(&VkWriteDescriptorSet {
            binding: 5,
            kind: VkWriteDescriptorSetKind::StorageBuffer {
                buffer: &bottom_as.geometry_info_buffer,
            },
        });
    });

    Ok((pool, sets))
}

fn create_and_record_command_buffers(
    pool: &VkCommandPool,
    swapchain: &VkSwapchain,
    sbt: &VkShaderBindingTable,
    pipeline_layout: &VkPipelineLayout,
    pipeline: &VkPipeline,
    descriptor_sets: &VkDescriptorSets,
    storage_images: &[ImageAndView],
) -> Result<Vec<VkCommandBuffer>> {
    log::debug!("Creating and recording command buffers");
    let buffers = pool
        .allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, swapchain.images.len() as _)?;

    for (index, buffer) in buffers.iter().enumerate() {
        let descriptor_set = &descriptor_sets.sets[index];
        let swapchain_image = &swapchain.images[index];
        let storage_image = &storage_images[index].image;

        buffer.begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;

        buffer.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);

        buffer.bind_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline_layout,
            descriptor_set,
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