#[cfg(feature = "ort-coreml")]
use ort::ep::CoreML;
#[cfg(feature = "ort-directml")]
use ort::ep::DirectML;
#[cfg(feature = "ort-rocm")]
use ort::ep::ROCm;
#[cfg(feature = "ort-tensorrt")]
use ort::ep::TensorRT;
#[cfg(feature = "ort-webgpu")]
use ort::ep::WebGPU;
use ort::ep::CPU;
#[cfg(feature = "ort-cuda")]
use ort::ep::CUDA;
#[cfg(feature = "ort-xnnpack")]
use ort::ep::XNNPACK;
#[cfg(feature = "ort-openvino")]
use ort::ep::OpenVINO;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;

use crate::accel::{get_ort_accelerator, OrtAccelerator};

#[cfg(feature = "ort-openvino")]
#[path = "openvino_plugin.rs"]
mod openvino_plugin;

/// Build the execution provider list based on the global accelerator preference.
fn execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
    let pref = get_ort_accelerator();
    let mut eps = Vec::new();

    match pref {
        OrtAccelerator::CpuOnly => {
            // CPU only — no GPU providers
        }
        OrtAccelerator::Cuda => {
            #[cfg(feature = "ort-cuda")]
            eps.push(CUDA::default().build());
            #[cfg(not(feature = "ort-cuda"))]
            log::warn!(
                "Accelerator set to CUDA but ort-cuda feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::TensorRt => {
            #[cfg(feature = "ort-tensorrt")]
            {
                eps.push(TensorRT::default().build());
                // CUDA as fallback for ops TensorRT doesn't support
                eps.push(CUDA::default().build());
            }
            #[cfg(not(feature = "ort-tensorrt"))]
            log::warn!(
                "Accelerator set to TensorRT but ort-tensorrt feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::DirectMl => {
            #[cfg(feature = "ort-directml")]
            eps.push(DirectML::default().build());
            #[cfg(not(feature = "ort-directml"))]
            log::warn!("Accelerator set to DirectML but ort-directml feature is not enabled; falling back to CPU");
        }
        OrtAccelerator::Rocm => {
            #[cfg(feature = "ort-rocm")]
            eps.push(ROCm::default().build());
            #[cfg(not(feature = "ort-rocm"))]
            log::warn!(
                "Accelerator set to ROCm but ort-rocm feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::CoreMl => {
            #[cfg(feature = "ort-coreml")]
            eps.push(CoreML::default().build());
            #[cfg(not(feature = "ort-coreml"))]
            log::warn!(
                "Accelerator set to CoreML but ort-coreml feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::WebGpu => {
            #[cfg(feature = "ort-webgpu")]
            eps.push(WebGPU::default().build());
            #[cfg(not(feature = "ort-webgpu"))]
            log::warn!(
                "Accelerator set to WebGPU but ort-webgpu feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::Xnnpack => {
            #[cfg(feature = "ort-xnnpack")]
            {
                let n = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                if let Some(nz) = core::num::NonZeroUsize::new(n) {
                    eps.push(XNNPACK::default().with_intra_op_num_threads(nz).build());
                } else {
                    eps.push(XNNPACK::default().build());
                }
            }
            #[cfg(not(feature = "ort-xnnpack"))]
            log::warn!(
                "Accelerator set to XNNPACK but ort-xnnpack feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::OpenVino => {
            #[cfg(feature = "ort-openvino")]
            {
                let mut ov = OpenVINO::default();
                if let Ok(dev) = std::env::var("HANDY_OPENVINO_DEVICE")
                    .or_else(|_| std::env::var("OPENVINO_DEVICE"))
                {
                    let dev = dev.trim();
                    if !dev.is_empty() {
                        log::info!("OpenVINO device_type override from env: {dev}");
                        ov = ov.with_device_type(dev.to_string());
                    }
                }
                match ov.is_available() {
                    Ok(true) => log::info!("OpenVINO EP available"),
                    Ok(false) => log::error!(
                        "OpenVINO EP is NOT available in this ONNX Runtime binary. \
                         Install OpenVINO Toolkit and a matching ORT build/plugin for Intel acceleration."
                    ),
                    Err(e) => log::error!("Failed to query OpenVINO EP availability: {e}"),
                }
                eps.push(ov.build());
            }
            #[cfg(not(feature = "ort-openvino"))]
            log::warn!(
                "Accelerator set to OpenVINO but ort-openvino feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::Npu => {
            #[cfg(feature = "ort-openvino")]
            {
                let ov = OpenVINO::default().with_device_type("NPU".to_string());
                match ov.is_available() {
                    Ok(true) => {
                        log::info!("OpenVINO EP available — targeting device_type=NPU");
                        eps.push(ov.build());
                    }
                    Ok(false) => {
                        log::error!(
                            "OpenVINO EP is NOT available in this ONNX Runtime binary. \
                             Intel NPU acceleration requires an ORT build (or plugin) that \
                             includes OpenVINO. Install OpenVINO Toolkit (winget install \
                             Intel.OpenVINOToolkit.2026.2.0) and use a matching ORT, or the \
                             NPU graph will stay flat and inference falls back to CPU."
                        );
                        eps.push(ov.build());
                    }
                    Err(e) => {
                        log::error!("Failed to query OpenVINO EP availability: {e}");
                        eps.push(ov.build());
                    }
                }
            }
            #[cfg(not(feature = "ort-openvino"))]
            log::warn!(
                "Accelerator set to NPU but ort-openvino feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::Auto => {
            #[cfg(feature = "ort-tensorrt")]
            eps.push(TensorRT::default().build());
            #[cfg(feature = "ort-cuda")]
            eps.push(CUDA::default().build());
            #[cfg(feature = "ort-rocm")]
            eps.push(ROCm::default().build());
            #[cfg(feature = "ort-openvino")]
            eps.push(OpenVINO::default().build());
            #[cfg(feature = "ort-coreml")]
            eps.push(CoreML::default().build());
        }
    }

    eps.push(CPU::default().build());
    eps
}

fn requires_sequential_session() -> bool {
    let pref = get_ort_accelerator();
    (pref == OrtAccelerator::DirectMl && cfg!(feature = "ort-directml"))
        || (pref == OrtAccelerator::WebGpu && cfg!(feature = "ort-webgpu"))
}

fn is_xnnpack_active() -> bool {
    let pref = get_ort_accelerator();
    pref == OrtAccelerator::Xnnpack && cfg!(feature = "ort-xnnpack")
}

fn build_session(
    path: &Path,
    intra_threads: Option<usize>,
    parallel_execution: bool,
) -> Result<Session, ort::Error> {
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

    if is_xnnpack_active() {
        builder = builder.with_intra_op_spinning(false)?;
        builder = builder.with_intra_threads(1)?;
    } else if let Some(n) = intra_threads {
        if n > 0 {
            builder = builder.with_intra_threads(n)?;
        }
    }

    let use_parallel = if requires_sequential_session() {
        false
    } else {
        parallel_execution
    };

    builder = builder.with_parallel_execution(use_parallel)?;

    if requires_sequential_session() {
        builder = builder.with_memory_pattern(false)?;
    }

    let pref = get_ort_accelerator();
    let session = {
        #[cfg(feature = "ort-openvino")]
        {
            let try_plugin = matches!(pref, OrtAccelerator::Npu | OrtAccelerator::OpenVino);
            if try_plugin {
                openvino_plugin::ensure_registered();
                match openvino_plugin::apply_devices(builder, pref == OrtAccelerator::Npu) {
                    Ok((b, true)) => {
                        log::info!("Session using OpenVINO EP via plugin device selection");
                        b.commit_from_file(path)?
                    }
                    Ok((b, false)) => {
                        log::warn!(
                            "OpenVINO plugin devices unavailable; falling back to classic EP list \
                             (likely no OpenVINO EP in this ORT binary)"
                        );
                        b.with_execution_providers(execution_providers())?
                            .commit_from_file(path)?
                    }
                    Err(e) => {
                        log::error!("OpenVINO plugin device selection failed: {e}; falling back");
                        let mut builder2 = Session::builder()?
                            .with_optimization_level(GraphOptimizationLevel::Level3)?;
                        if is_xnnpack_active() {
                            builder2 = builder2.with_intra_op_spinning(false)?;
                            builder2 = builder2.with_intra_threads(1)?;
                        } else if let Some(n) = intra_threads {
                            if n > 0 {
                                builder2 = builder2.with_intra_threads(n)?;
                            }
                        }
                        let use_parallel = if requires_sequential_session() {
                            false
                        } else {
                            parallel_execution
                        };
                        builder2 = builder2.with_parallel_execution(use_parallel)?;
                        if requires_sequential_session() {
                            builder2 = builder2.with_memory_pattern(false)?;
                        }
                        builder2
                            .with_execution_providers(execution_providers())?
                            .commit_from_file(path)?
                    }
                }
            } else {
                builder
                    .with_execution_providers(execution_providers())?
                    .commit_from_file(path)?
            }
        }
        #[cfg(not(feature = "ort-openvino"))]
        {
            let _ = pref;
            builder
                .with_execution_providers(execution_providers())?
                .commit_from_file(path)?
        }
    };

    for input in session.inputs() {
        log::info!(
            "Model input: name={}, type={:?}",
            input.name(),
            input.dtype()
        );
    }
    for output in session.outputs() {
        log::info!(
            "Model output: name={}, type={:?}",
            output.name(),
            output.dtype()
        );
    }

    Ok(session)
}

pub fn create_session(path: &Path) -> Result<Session, ort::Error> {
    build_session(path, None, true)
}

pub fn create_session_with_threads(path: &Path, num_threads: usize) -> Result<Session, ort::Error> {
    build_session(path, Some(num_threads), true)
}

pub fn resolve_model_path(
    dir: &Path,
    name: &str,
    quantization: &super::Quantization,
) -> std::path::PathBuf {
    let suffix = match quantization {
        super::Quantization::FP32 => None,
        super::Quantization::FP16 => Some("fp16"),
        super::Quantization::Int8 => Some("int8"),
        super::Quantization::Int4 => Some("int4"),
    };

    if let Some(suffix) = suffix {
        let path = dir.join(format!("{}.{}.onnx", name, suffix));
        if path.exists() {
            log::info!("Loading {} model: {}", suffix, path.display());
            return path;
        }
        log::warn!(
            "{} model not found at {}, falling back to {}.onnx",
            suffix,
            path.display(),
            name
        );
    }

    dir.join(format!("{}.onnx", name))
}

pub fn read_metadata_str(session: &Session, key: &str) -> Result<Option<String>, ort::Error> {
    let meta = session.metadata()?;
    Ok(meta.custom(key).filter(|s| !s.is_empty()))
}

pub fn read_metadata_i32(
    session: &Session,
    key: &str,
    default: Option<i32>,
) -> Result<Option<i32>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key).map_err(|e| {
        crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e))
    })?;
    match str_val {
        Some(v) => Ok(Some(v.parse::<i32>().map_err(|e| {
            crate::TranscribeError::Config(format!("failed to parse '{}': {}", key, e))
        })?)),
        None => Ok(default),
    }
}

pub fn read_metadata_float_vec(
    session: &Session,
    key: &str,
) -> Result<Option<Vec<f32>>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key).map_err(|e| {
        crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e))
    })?;
    match str_val {
        Some(v) => {
            let floats: Result<Vec<f32>, _> =
                v.split(',').map(|s| s.trim().parse::<f32>()).collect();
            Ok(Some(floats.map_err(|e| {
                crate::TranscribeError::Config(format!(
                    "failed to parse floats in '{}': {}",
                    key, e
                ))
            })?))
        }
        None => Ok(None),
    }
}
