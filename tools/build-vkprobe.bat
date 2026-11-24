@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /O2 /EHsc /I"C:\VulkanSDK\1.4.363.0\Include" "%~dp0vkprobe.c" "C:\VulkanSDK\1.4.363.0\Lib\vulkan-1.lib" /Fe:"%~dp0vkprobe.exe" /Fo:"%~dp0vkprobe.obj"
