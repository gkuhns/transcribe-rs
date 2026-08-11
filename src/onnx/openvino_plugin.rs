//! OpenVINO EP plugin registration and device selection for Intel NPU/GPU/CPU.
//!
//! Microsoft/pyke prebuilt ORT does not include OpenVINO. At runtime we try to
//! register Intel's `onnxruntime_providers_openvino.dll` plugin and select
//! devices via the ORT V2 EP device API.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::memory::DeviceType;

/// Keep the EP library handle alive for the process lifetime so it is not unregistered.
static OPENVINO_EP_LIB: OnceLock<Result<ort::ep::ExecutionProviderLibrary, String>> = OnceLock::new();

fn candidate_plugin_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for key in ["HANDY_OPENVINO_EP_LIBRARY", "ORT_OPENVINO_EP_LIBRARY"] {
        if let Ok(v) = std::env::var(key) {
            let t = v.trim();
            if !t.is_empty() {
                paths.push(PathBuf::from(t));
            }
        }
    }
    // Common locations after winget / archive OpenVINO install or manual plugin drop.
    let extras = [
        r"C:\Program Files\Intel\OpenVINO\onnxruntime_providers_openvino.dll",
        r"C:\Program Files (x86)\Intel\OpenVINO\onnxruntime_providers_openvino.dll",
        r"C:\Program Files\onnxruntime-ep-openvino\onnxruntime_providers_openvino.dll",
        "onnxruntime_providers_openvino.dll",
        "plugins/onnxruntime_providers_openvino.dll",
    ];
    for e in extras {
        paths.push(PathBuf::from(e));
    }
    // Alongside the running executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("onnxruntime_providers_openvino.dll"));
            paths.push(dir.join("plugins").join("onnxruntime_providers_openvino.dll"));
        }
    }
    paths
}

/// Attempt once to register Intel's OpenVINO EP plugin with the current ORT environment.
/// Safe to call repeatedly; subsequent calls reuse the first result.
pub fn ensure_registered() -> bool {
    let result = OPENVINO_EP_LIB.get_or_init(|| {
        // Ensure environment exists
        let env = Environment::current().map_err(|e| format!("Environment::current failed: {e}"))?;

        let mut last_err = String::from("no candidate plugin paths found");
        for path in candidate_plugin_paths() {
            if !path.exists() {
                continue;
            }
            log::info!("Attempting to register OpenVINO EP plugin from {}", path.display());
            match env.register_ep_library("openvino_ep", &path) {
                Ok(handle) => {
                    log::info!("Registered OpenVINO EP plugin from {}", path.display());
                    return Ok(handle);
                }
                Err(e) => {
                    last_err = format!("{}: {e}", path.display());
                    log::warn!("Failed to register OpenVINO EP plugin from {}: {e}", path.display());
                }
            }
        }
        Err(last_err)
    });

    match result {
        Ok(_) => true,
        Err(e) => {
            log::error!(
                "OpenVINO EP plugin not registered ({e}). Place onnxruntime_providers_openvino.dll \
                 next to the app or set HANDY_OPENVINO_EP_LIBRARY. \
                 Install OpenVINO Runtime: winget install --id Intel.OpenVINOToolkit.2026.2.0 -e"
            );
            false
        }
    }
}

/// Apply OpenVINO (optionally NPU-only) devices to a session builder via the V2 API.
/// Returns true if at least one device was applied.
pub fn apply_devices(
    builder: ort::session::builder::SessionBuilder,
    prefer_npu: bool,
) -> Result<(ort::session::builder::SessionBuilder, bool), ort::Error> {
    ensure_registered();
    let env = Environment::current()?;
    let devices: Vec<_> = env
        .devices()
        .filter(|dev| {
            let Ok(ep_name) = dev.ep() else { return false };
            let ep_l = ep_name.to_ascii_lowercase();
            if !(ep_l.contains("openvino") || ep_l.contains("ov")) {
                return false;
            }
            let ty = dev.hardware_device().ty();
            if prefer_npu {
                ty == DeviceType::NPU
            } else {
                true
            }
        })
        .collect();

    if devices.is_empty() {
        log::warn!(
            "No OpenVINO EP devices discovered after plugin registration (prefer_npu={prefer_npu})"
        );
        return Ok((builder, false));
    }

    for d in &devices {
        if let (Ok(ep), hw) = (d.ep(), d.hardware_device()) {
            log::info!(
                "Selecting OpenVINO device: ep={ep} hw={:?} id={}",
                hw.ty(),
                hw.id()
            );
        }
    }

    let options = vec![(
        "OpenVINOExecutionProvider.device_type".to_string(),
        if prefer_npu {
            "NPU".to_string()
        } else {
            "AUTO".to_string()
        },
    )];
    let builder = builder.with_devices(devices, Some(&options))?;
    Ok((builder, true))
}
