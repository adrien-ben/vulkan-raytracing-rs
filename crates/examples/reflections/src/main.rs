use anyhow::Result;
use ash::vk::{self, Packed24_8};
use glam::{vec3, Mat4};
use gltf::Vertex;
use gpu_allocator::MemoryLocation;
use gui::{
    imgui::{self, DrawData},
    imgui_rs_vulkan_renderer::Renderer,
    GuiContext,
};
use simple_logger::SimpleLogger;
use std::{
    mem::{size_of, size_of_val},
    time::Instant,
};
use vulkan::utils::*;
use vulkan::*;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const IN_FLIGHT_FRAMES: u32 = 2;
const APP_NAME: &str = "Reflections";

const MODEL_PATH: &str = "./assets/models/reflections.glb";
const EYE_POS: [f32; 3] = [-2.0, 1.5, 2.0];
const EYE_TARGET: [f32; 3] = [0.0, 1.0, 0.0];
const MAX_DEPTH: u32 = 10;

fn main() -> Result<()> {
    SimpleLogger::default().env().init()?;

    let (window, event_loop) = create_window();
    let mut app = App::new(&window)?;
    let mut gui_context = GuiContext::new(
        &app.context,
        &app.command_pool,
        &app.render_pass,
        &window,
        IN_FLIGHT_FRAMES as _,
    )?;
    let mut is_swapchain_dirty = false;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        gui_context.handle_event(&window, &event);

        match event {
            Event::NewEvents(_) => {
                let now = Instant::now();
                gui_context.update_delta_time(now - last_frame);
                last_frame = now;
            }
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
                        gui_context
                            .set_render_pass(&app.render_pass)
                            .expect("Failed to set gui render pass");
                    } else {
                        return;
                    }
                }

                is_swapchain_dirty = app.draw(&window, &mut gui_context).expect("Failed to tick");
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

struct Model {
    gltf: gltf::Model,
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    transform_buffer: VkBuffer,
    images: Vec<VkImage>,
    views: Vec<VkImageView>,
    samplers: Vec<VkSampler>,
    textures: Vec<(usize, usize)>,
}

struct BottomAS {
    inner: VkAccelerationStructure,
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

struct Camera {
    position: [f32; 3],
    target: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Light {
    direction: [f32; 3],
    color: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SceneUBO {
    inverted_view: Mat4,
    inverted_proj: Mat4,
    light_direction: [f32; 4],
    light_color: [f32; 4],
    max_depth: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GeometryInfo {
    transform: Mat4,
    base_color: [f32; 4],
    base_color_texture_index: i32,
    metallic_factor: f32,
    vertex_offset: u32,
    index_offset: u32,
}

struct App {
    swapchain: VkSwapchain,
    render_pass: VkRenderPass,
    framebuffers: Vec<VkFramebuffer>,
    command_pool: VkCommandPool,
    pipeline_res: PipelineRes,
    camera: Camera,
    light: Light,
    max_depth: u32,
    ubo_buffer: VkBuffer,
    _model: Model,
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

        let command_pool = context.create_command_pool(
            context.graphics_queue_family,
            Some(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
        )?;

        let swapchain = VkSwapchain::new(&context, WIDTH, HEIGHT)?;

        let render_pass = create_render_pass(&context, &swapchain)?;

        let framebuffers = swapchain.get_framebuffers(&render_pass)?;

        let camera = Camera {
            position: EYE_POS,
            target: EYE_TARGET,
        };
        let light = Light {
            direction: [-2.0, -1.0, -2.0],
            color: [1.0; 3],
        };
        let ubo_buffer = context.create_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            size_of::<SceneUBO>() as _,
        )?;

        let model = create_model(&context)?;

        let bottom_as = create_bottom_as(&mut context, &model)?;

        let top_as = create_top_as(&mut context, &bottom_as)?;

        let storage_images = create_storage_images(
            &mut context,
            swapchain.format,
            swapchain.extent,
            swapchain.images.len(),
        )?;

        let pipeline_res = create_pipeline(&context, &model)?;

        let sbt = context.create_shader_binding_table(&pipeline_res.pipeline)?;

        let descriptor_res = create_descriptor_sets(
            &context,
            &pipeline_res,
            &model,
            &bottom_as,
            &top_as,
            &storage_images,
            &ubo_buffer,
        )?;

        let command_buffers = create_command_buffers(&command_pool, &swapchain)?;

        let in_flight_frames = InFlightFrames::new(&context, IN_FLIGHT_FRAMES)?;

        Ok(Self {
            context,
            command_pool,
            swapchain,
            render_pass,
            framebuffers,
            pipeline_res,
            camera,
            light,
            max_depth: MAX_DEPTH,
            ubo_buffer,
            _model: model,
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

        // Swapchain and dependent resources
        self.framebuffers.clear();
        self.swapchain.resize(&self.context, width, height)?;

        let render_pass = create_render_pass(&self.context, &self.swapchain)?;
        let _ = std::mem::replace(&mut self.render_pass, render_pass);

        let framebuffers = self.swapchain.get_framebuffers(&self.render_pass)?;
        let _ = std::mem::replace(&mut self.framebuffers, framebuffers);

        // Recreate storage image for RT and update descriptor set
        let storage_images = create_storage_images(
            &mut self.context,
            self.swapchain.format,
            self.swapchain.extent,
            self.swapchain.images.len(),
        )?;

        storage_images.iter().enumerate().for_each(|(index, img)| {
            let set = &self.descriptor_res.dynamic_sets[index];

            set.update_one(VkWriteDescriptorSet {
                binding: 1,
                kind: VkWriteDescriptorSetKind::StorageImage {
                    layout: vk::ImageLayout::GENERAL,
                    view: &img.view,
                },
            });
        });

        let _ = std::mem::replace(&mut self.storage_images, storage_images);

        Ok(())
    }

    fn draw(&mut self, window: &Window, gui_context: &mut GuiContext) -> Result<bool> {
        // Generate UI

        gui_context
            .platform
            .prepare_frame(gui_context.imgui.io_mut(), window)?;
        let ui = gui_context.imgui.frame();

        imgui::Window::new("Vulkan RT")
            .size([300.0, 400.0], imgui::Condition::FirstUseEver)
            .build(&ui, || {
                // RT controls
                ui.text_wrapped("Rays");
                let mut max_depth = self.max_depth as _;
                ui.input_int("max depth", &mut max_depth).build();
                self.max_depth = max_depth.max(1) as _;

                // Cam controls
                ui.text_wrapped("Camera");
                ui.separator();

                ui.input_float3("position", &mut self.camera.position)
                    .build();
                ui.input_float3("target", &mut self.camera.target).build();

                // Light control
                ui.text_wrapped("Light");
                ui.separator();

                ui.input_float3("direction", &mut self.light.direction)
                    .build();

                imgui::ColorPicker::new("color", &mut self.light.color)
                    .display_rgb(true)
                    .build(&ui);
            });

        gui_context.platform.prepare_render(&ui, window);
        let draw_data = ui.render();

        // Drawing the frame
        self.in_flight_frames.next();
        self.in_flight_frames.fence().wait(None)?;

        let next_image_result = self.swapchain.acquire_next_image(
            std::u64::MAX,
            self.in_flight_frames.image_available_semaphore(),
        );
        let image_index = match next_image_result {
            Ok(AcquiredImage { index, .. }) => index as usize,
            Err(err) => match err.downcast_ref::<vk::Result>() {
                Some(&vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                _ => panic!("Error while acquiring next image. Cause: {}", err),
            },
        };

        self.update_ubo_buffer()?;

        self.in_flight_frames.fence().reset()?;

        let command_buffer = &self.command_buffers[image_index];

        self.record_command_buffer(
            command_buffer,
            image_index,
            &mut gui_context.renderer,
            draw_data,
        )?;

        self.context.graphics_queue.submit(
            command_buffer,
            Some(self.in_flight_frames.image_available_semaphore()),
            Some(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
            Some(self.in_flight_frames.render_finished_semaphore()),
            self.in_flight_frames.fence(),
        )?;

        let signal_semaphores = [self.in_flight_frames.render_finished_semaphore()];
        let present_result = self.swapchain.queue_present(
            image_index as _,
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

    fn update_ubo_buffer(&self) -> Result<()> {
        let view = Mat4::look_at_rh(
            self.camera.position.into(),
            self.camera.target.into(),
            vec3(0.0, 1.0, 0.0),
        );
        let inverted_view = view.inverse();

        let width = self.swapchain.extent.width as f32;
        let height = self.swapchain.extent.height as f32;

        let proj = Mat4::perspective_infinite_rh(60f32.to_radians(), width / height, 0.1);
        let inverted_proj = proj.inverse();

        let light_direction = [
            self.light.direction[0],
            self.light.direction[1],
            self.light.direction[2],
            0.0,
        ];
        let light_color = [
            self.light.color[0],
            self.light.color[1],
            self.light.color[2],
            0.0,
        ];

        let scene_ubo = SceneUBO {
            inverted_view,
            inverted_proj,
            light_direction,
            light_color,
            max_depth: self.max_depth,
        };

        self.ubo_buffer.copy_data_to_buffer(&[scene_ubo])?;

        Ok(())
    }

    fn record_command_buffer(
        &self,
        buffer: &VkCommandBuffer,
        image_index: usize,
        gui_renderer: &mut Renderer,
        draw_data: &DrawData,
    ) -> Result<()> {
        let swapchain_image = &self.swapchain.images[image_index];
        let framebuffer = &self.framebuffers[image_index];
        let static_set = &self.descriptor_res.static_set;
        let dynamic_set = &self.descriptor_res.dynamic_sets[image_index];
        let storage_image = &self.storage_images[image_index];

        let storage_image = &storage_image.image;

        buffer.reset()?;

        buffer.begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;

        // Ray Tracing
        buffer.bind_pipeline(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            &self.pipeline_res.pipeline,
        );

        buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            &self.pipeline_res.pipeline_layout,
            0,
            &[static_set, dynamic_set],
        );

        buffer.trace_rays(
            &self.sbt,
            swapchain_image.extent.width,
            swapchain_image.extent.height,
        );

        // Copy ray tracing result into swapchain
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

        // Gui pass
        buffer.begin_render_pass(&self.render_pass, framebuffer);

        gui_renderer.cmd_draw(buffer.inner, draw_data)?;

        buffer.end_render_pass();

        buffer.end()?;

        Ok(())
    }

    pub fn wait_for_gpu(&self) -> Result<()> {
        self.context.device_wait_idle()
    }
}

fn create_render_pass(context: &VkContext, swapchain: &VkSwapchain) -> Result<VkRenderPass> {
    let attachment_descs = [vk::AttachmentDescription::builder()
        .format(swapchain.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let color_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let subpass_descs = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .build()];

    let subpass_deps = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    context.create_render_pass(&render_pass_info)
}

fn create_model(context: &VkContext) -> Result<Model> {
    let model = gltf::load_file(MODEL_PATH)?;
    let vertices = model.vertices.as_slice();
    let indices = model.indices.as_slice();

    let vertex_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        vertices,
    )?;

    let index_buffer = create_gpu_only_buffer_from_data(
        context,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        indices,
    )?;

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

    let mut images = vec![];
    let mut views = vec![];

    model.images.iter().try_for_each::<_, Result<_>>(|i| {
        let width = i.width;
        let height = i.height;
        let pixels = i.pixels.as_slice();

        let staging = context.create_buffer(
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            size_of_val(pixels) as _,
        )?;

        staging.copy_data_to_buffer(pixels)?;

        let image = context.create_image(
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            vk::Format::R8G8B8A8_SRGB,
            width,
            height,
        )?;

        context.execute_one_time_commands(|cmd| {
            cmd.transition_layout(
                &image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            );

            cmd.copy_buffer_to_image(&staging, &image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

            cmd.transition_layout(
                &image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );
        })?;

        let view = image.create_image_view()?;

        images.push(image);
        views.push(view);

        Ok(())
    })?;

    // Dummy textures
    if images.is_empty() {
        let image = context.create_image(
            vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            vk::Format::R8G8B8A8_SRGB,
            1,
            1,
        )?;

        context.execute_one_time_commands(|cmd| {
            cmd.transition_layout(
                &image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );
        })?;

        let view = image.create_image_view()?;

        images.push(image);
        views.push(view);
    }

    let mut samplers = model
        .samplers
        .iter()
        .map(|s| {
            let sampler_info = map_gltf_sampler(s);
            context.create_sampler(&sampler_info)
        })
        .collect::<Result<Vec<_>>>()?;

    // Dummy sampler
    if samplers.is_empty() {
        let sampler_info = vk::SamplerCreateInfo::builder();
        let sampler = context.create_sampler(&sampler_info)?;
        samplers.push(sampler);
    }

    let mut textures = model
        .textures
        .iter()
        .map(|t| (t.image_index, t.sampler_index))
        .collect::<Vec<_>>();

    // Dummy texture
    if textures.is_empty() {
        textures.push((0, 0));
    }

    Ok(Model {
        gltf: model,
        vertex_buffer,
        index_buffer,
        transform_buffer,
        images,
        views,
        samplers,
        textures,
    })
}

fn map_gltf_sampler<'a>(sampler: &gltf::Sampler) -> vk::SamplerCreateInfoBuilder<'a> {
    let mag_filter = match sampler.mag_filter {
        gltf::MagFilter::Linear => vk::Filter::LINEAR,
        gltf::MagFilter::Nearest => vk::Filter::NEAREST,
    };

    let min_filter = match sampler.min_filter {
        gltf::MinFilter::Linear
        | gltf::MinFilter::LinearMipmapLinear
        | gltf::MinFilter::LinearMipmapNearest => vk::Filter::LINEAR,
        gltf::MinFilter::Nearest
        | gltf::MinFilter::NearestMipmapLinear
        | gltf::MinFilter::NearestMipmapNearest => vk::Filter::NEAREST,
    };

    vk::SamplerCreateInfo::builder()
        .mag_filter(mag_filter)
        .min_filter(min_filter)
}

fn create_bottom_as(context: &mut VkContext, model: &Model) -> Result<BottomAS> {
    let vertex_buffer_addr = model.vertex_buffer.get_device_address();

    let index_buffer_addr = model.index_buffer.get_device_address();

    let transform_buffer_addr = model.transform_buffer.get_device_address();

    let as_geo_triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_buffer_addr,
        })
        .vertex_stride(size_of::<Vertex>() as _)
        .max_vertex(model.gltf.vertices.len() as _)
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

    for (node_index, node) in model.gltf.nodes.iter().enumerate() {
        let mesh = node.mesh;

        let primitive_count = (mesh.index_count / 3) as u32;

        geometry_infos.push(GeometryInfo {
            transform: Mat4::from_cols_array_2d(&node.transform),
            base_color: mesh.material.base_color,
            base_color_texture_index: mesh
                .material
                .base_color_texture_index
                .map_or(-1, |i| i as _),
            metallic_factor: mesh.material.metallic_factor,
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

fn create_pipeline(context: &VkContext, model: &Model) -> Result<PipelineRes> {
    // descriptor and pipeline layouts
    let static_layout_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
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
        // Textures
        vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(model.images.len() as _)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
    ];

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
            source: &include_bytes!("../shaders/raygen.rgen.spv")[..],
            stage: vk::ShaderStageFlags::RAYGEN_KHR,
            group: VkRTShaderGroup::RayGen,
        },
        VkRTShaderCreateInfo {
            source: &include_bytes!("../shaders/miss.rmiss.spv")[..],
            stage: vk::ShaderStageFlags::MISS_KHR,
            group: VkRTShaderGroup::Miss,
        },
        VkRTShaderCreateInfo {
            source: &include_bytes!("../shaders/shadow.rmiss.spv")[..],
            stage: vk::ShaderStageFlags::MISS_KHR,
            group: VkRTShaderGroup::Miss,
        },
        VkRTShaderCreateInfo {
            source: &include_bytes!("../shaders/closesthit.rchit.spv")[..],
            stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            group: VkRTShaderGroup::ClosestHit,
        },
    ];

    let pipeline_create_info = VkRTPipelineCreateInfo {
        shaders: &shaders_create_info,
        max_ray_recursion_depth: 2,
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
    model: &Model,
    bottom_as: &BottomAS,
    top_as: &TopAS,
    storage_imgs: &[ImageAndView],
    ubo_buffer: &VkBuffer,
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
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(model.images.len() as _)
            .build(),
    ];

    let pool = context.create_descriptor_pool(set_count + 1, &pool_sizes)?;

    let static_set = pool.allocate_set(&pipeline_res.static_dsl)?;
    let dynamic_sets = pool.allocate_sets(&pipeline_res.dynamic_dsl, set_count)?;

    static_set.update(&[
        VkWriteDescriptorSet {
            binding: 0,
            kind: VkWriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: &top_as.inner,
            },
        },
        VkWriteDescriptorSet {
            binding: 2,
            kind: VkWriteDescriptorSetKind::UniformBuffer { buffer: ubo_buffer },
        },
        VkWriteDescriptorSet {
            binding: 3,
            kind: VkWriteDescriptorSetKind::StorageBuffer {
                buffer: &model.vertex_buffer,
            },
        },
        VkWriteDescriptorSet {
            binding: 4,
            kind: VkWriteDescriptorSetKind::StorageBuffer {
                buffer: &model.index_buffer,
            },
        },
        VkWriteDescriptorSet {
            binding: 5,
            kind: VkWriteDescriptorSetKind::StorageBuffer {
                buffer: &bottom_as.geometry_info_buffer,
            },
        },
    ]);

    for (image_index, sampler_index) in model.textures.iter() {
        let view = &model.views[*image_index];
        let sampler = &model.samplers[*sampler_index];

        static_set.update_one(VkWriteDescriptorSet {
            binding: 6,
            kind: VkWriteDescriptorSetKind::CombinedImageSampler {
                view,
                sampler,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        })
    }

    dynamic_sets.iter().enumerate().for_each(|(index, set)| {
        set.update_one(VkWriteDescriptorSet {
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

fn create_command_buffers(
    pool: &VkCommandPool,
    swapchain: &VkSwapchain,
) -> Result<Vec<VkCommandBuffer>> {
    pool.allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, swapchain.images.len() as _)
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

    fn next(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();
    }

    fn image_available_semaphore(&self) -> &VkSemaphore {
        &self.sync_objects[self.current_frame].image_available_semaphore
    }

    fn render_finished_semaphore(&self) -> &VkSemaphore {
        &self.sync_objects[self.current_frame].render_finished_semaphore
    }

    fn fence(&self) -> &VkFence {
        &self.sync_objects[self.current_frame].fence
    }
}

struct SyncObjects {
    image_available_semaphore: VkSemaphore,
    render_finished_semaphore: VkSemaphore,
    fence: VkFence,
}
