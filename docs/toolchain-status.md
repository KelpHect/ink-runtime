# Toolchain status

Host: Windows 11. Set up for a Vulkan-compute inference runtime (C/C++ build).
Recorded by the setup agent. Machine policy asked for toolchains under
`C:\dev-overflow\<tool>`; winget installs went to their standard locations
instead, which is allowed — paths are recorded below.

## Summary

| Tool | Version | Status | Location |
| ---- | ------- | ------ | -------- |
| CMake | 4.4.4 | Installed this session (winget `Kitware.CMake`, portable) | `C:\Users\KellHect\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.4-windows-x86_64\bin\cmake.exe` |
| Vulkan SDK | 1.4.363.0 | Installed this session (winget `KhronosGroup.VulkanSDK`, elevated OK) | `C:\VulkanSDK\1.4.363.0` |
| glslc | shaderc v2026.4 (glslang 11.1.0-1566-ge1b562a8) | In SDK | `C:\VulkanSDK\1.4.363.0\Bin\glslc.exe` |
| MSVC cl | 19.44.35228 (toolset 14.44.35207) | Already installed (VS 2022 BuildTools, VCTools workload) | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` |
| Windows SDK | 10.0.26100.0 | Already installed | `C:\Program Files (x86)\Windows Kits\10` |
| vulkaninfo | SDK 1.4.363 | Driver copy + SDK copy | `C:\Windows\System32\vulkaninfo.exe`, `C:\VulkanSDK\1.4.363.0\Bin\vulkaninfoSDK.exe` |

## Details

### CMake (new)

`winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements --silent --disable-interactivity`
succeeded. It is the portable zip package: no admin needed, unpacked under the
user WinGet packages folder. `cmake`, `ctest`, `cpack`, `cmake-gui` aliases were
added to the **user PATH — only new shells see them**. In an existing shell use
the full path above. `cmake --version` → `cmake version 4.4.4`.

### Vulkan SDK (new)

`winget install --id KhronosGroup.VulkanSDK -e --accept-source-agreements --accept-package-agreements --silent --disable-interactivity`
succeeded (installer requested admin; elevation went through). Installs to
`C:\VulkanSDK\1.4.363.0` — includes `Bin\glslc.exe`, `Bin\glslangValidator.exe`,
`Bin\vulkaninfoSDK.exe`, `Include\`, `Lib\` (vulkan-1.lib, volk-compatible
loader), and the validation layers. Dependency `Microsoft.VCRedist.2015+.x64`
was pulled in automatically. Sets `VULKAN_SDK` and PATH for new shells.

`glslc --version`:
shaderc v2026.4 / spirv-tools v2026.4 / glslang 11.1.0-1566-ge1b562a8.

### MSVC / VS 2022 BuildTools (pre-existing)

`winget install --id Microsoft.VisualStudio.2022.BuildTools ... --override "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
reported "Found an existing package already installed. Trying to upgrade…" and
the bootstrapper exited with code 1 (winget exit 6). **The failure was the
upgrade step, not the toolchain** — a functional BuildTools install was already
on the machine and was not harmed.

Verified working end-to-end: `vcvars64.bat` initializes the x64 dev shell
("Visual Studio 2022 Developer Command Prompt v17.14.39"), and a test C file
compiled with `cl`, linked with `link`, and ran correctly.

- Instance root: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
- Dev shell init: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`
- Also on this machine (not needed, listed for completeness):
  `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` (VS 2026 BT)
  and `C:\Program Files\Microsoft Visual Studio\18\Community` (VS 2026
  Community); all three carry `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`.

If a newer 2022 BuildTools build is ever required, run `winget upgrade
Microsoft.VisualStudio.2022.BuildTools` interactively (approve UAC) or use the
Visual Studio Installer GUI. Not needed for the runtime build.

### GPU / Vulkan

`vulkaninfo --summary` (full capture: `research/probe-vulkan-summary.txt`):

- Instance version 1.4.363 (SDK layers present, incl. `VK_LAYER_KHRONOS_validation`)
- GPU0: **AMD Radeon RX 9070 XT** (vendorID 0x1002, deviceID 0x7550, discrete)
- apiVersion 1.4.359, driverID `DRIVER_ID_AMD_PROPRIETARY`, driverInfo `26.10.1 (LLPC)`

## Gotchas

- **PATH lag**: cmake and the Vulkan SDK binaries are on PATH only in shells
  started *after* the installs. Scripts/agents in an older shell must use the
  absolute paths above.
- **`link` name collision**: `C:\Program Files\Git\usr\bin\link.exe` exists on a
  Git Bash PATH. Inside the vcvars64 dev shell the MSVC `link.exe` wins; in a
  plain bash shell call the MSVC path explicitly or initialize vcvars first.
- **Batch-file escaping**: when driving `cl`/vcvars from Git Bash, write `.bat`
  helpers with a plain file writer. `printf '...\\v...'` mangles `\v`, `\2` etc.
  into control characters (that was the sole cause of a spurious
  "path not found" during verification — the VS install itself is healthy).
