@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /O2 /EHsc /I"C:\VulkanSDK\1.4.363.0\Include" "%~dp0heapprobe.c" "C:\VulkanSDK\1.4.363.0\Lib\vulkan-1.lib" /Fe:"%~dp0heapprobe.exe" /Fo:"%~dp0heapprobe.obj"
