//! OpenVINO EP plugin registration and device selection for Intel NPU/GPU/CPU.
//!
//! Microsoft/pyke prebuilt ORT does not include OpenVINO. At runtime we try to
//! register Intel's `onnxruntime_providers_openvino.dll` plugin and select
//! devices via the ORT V2 EP device API.
//!
//! The plugin MUST NOT sit next to onnxruntime.dll. ORT auto-loads sibling
//! provider DLLs, which then hijacks Silero VAD and CPU decoder graphs through
//! the OpenVINO ONNX frontend (If-13 / If-16 conversion failures).

use std::path::PathBuf;
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
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("openvino-ep").join("onnxruntime_providers_openvino.dll"));
            paths.push(dir.join("plugins").join("onnxruntime_providers_openvino.dll"));
        }
    }
    paths
}

fn prepend_plugin_dir_to_path(plugin: &std::path::Path) {
    if let Some(dir) = plugin.parent() {
        let extra = dir.display().to_string();
        let key = "PATH";
        let joined = match std::env::var_os(key) {
            Some(existing) => {
                let mut combined = std::ffi::OsString::from(&extra);
                combined.push(";");
                combined.push(existing);
                combined
            }
            None => std::ffi::OsString::from(extra),
        };
        unsafe { std::env::set_var(key, joined) };
        log::info!("Prepended OpenVINO EP dir to PATH: {}", dir.display());
    }
}

pub fn ensure_registered() -> bool {
    let result = OPENVINO_EP_LIB.get_or_init(|| {
        let env = Environment::current().map_err(|e| format!("Environment::current failed: {e}"))?;

        let mut last_err = String::from("no candidate plugin paths found");
        for path in candidate_plugin_paths() {
            if !path.exists() {
                continue;
            }
            prepend_plugin_dir_to_path(&path);
            log::info!("Attempting to register OpenVINO EP plugin from {}", path.display());
            match env.register_ep_library("openvino_ep", &path) {
                Ok(handle) => {
                    log::info!("Registered OpenVINO EP plugin from {}", path.display());
                    return Ok(handle);
                }
                Err(e) => {
                    last_err = format!("{path}: {e}", path = path.display());
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
                "OpenVINO EP plugin not registered ({e}). Place the DLL under openvino-ep/ next to the app \
                 or set HANDY_OPENVINO_EP_LIBRARY."
            );
            false
        }
    }
}

/// Apply OpenVINO (optionally NPU-only) devices to a session builder via the V2 API.
/// Returns true if at least one device was applied.
///
/// Attaching both `OpenVINOExecutionProvider` and `OpenVINOExecutionProvider.AUTO`
/// at once has aborted native graph compile on Intel NPU. We keep a single device.
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
            let ty = dev.ty();
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

    let mut primary = Vec::new();
    let mut auto_devs = Vec::new();
    for d in devices {
        let ep = d.ep().map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        if ep.contains("auto") {
            auto_devs.push(d);
        } else {
            primary.push(d);
        }
    }
    let mut selected = if !primary.is_empty() { primary } else { auto_devs };
    selected.truncate(1);

    for d in &selected {
        if let Ok(ep) = d.ep() {
            log::info!(
                "Selecting OpenVINO device: ep={ep} hw={:?} id={}",
                d.ty(),
                d.id()
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
    let builder = builder.with_devices(selected, Some(&options))?;
    Ok((builder, true))
}
