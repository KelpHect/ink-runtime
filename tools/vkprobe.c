/* vkprobe.c — enumerate VK_KHR_cooperative_matrix configs + queue/memory on the
   first physical device. Compile (once SDK installed):
     cl /EHsc /I"%VULKAN_SDK%\Include" tools\vkprobe.c "%VULKAN_SDK%\Lib\vulkan-1.lib" /Fe:vkprobe.exe
   or:  gcc tools/vkprobe.c -o vkprobe -lvulkan -I"$VULKAN_SDK/Include" -L"$VULKAN_SDK/Lib" */
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

static const char* ctype(VkComponentTypeKHR t){
    switch(t){
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bf16";
    case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: return "f8e4m3";
    case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT: return "f8e5m2";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "i8";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "i16";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "i32";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    case VK_COMPONENT_TYPE_SINT64_KHR: return "i64";
    case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
    default: return "?";
    }
}
static const char* scope(VkScopeKHR s){
    switch(s){
    case VK_SCOPE_DEVICE_KHR: return "device";
    case VK_SCOPE_WORKGROUP_KHR: return "workgroup";
    case VK_SCOPE_SUBGROUP_KHR: return "subgroup";
    case VK_SCOPE_QUEUE_FAMILY_KHR: return "qfamily";
    default: return "?";
    }
}

int main(void){
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"vkprobe",1,"",0,VK_API_VERSION_1_3};
    const char* exts[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&app,0,0,1,exts};
    VkInstance inst;
    if(vkCreateInstance(&ici,0,&inst)!=VK_SUCCESS){puts("vkCreateInstance failed");return 1;}

    uint32_t ndev=0; vkEnumeratePhysicalDevices(inst,&ndev,0);
    if(!ndev){puts("no device");return 1;}
    VkPhysicalDevice* devs = malloc(ndev*sizeof*devs);
    vkEnumeratePhysicalDevices(inst,&ndev,devs);
    VkPhysicalDevice dev = devs[0];

    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vkGetPhysicalDeviceProperties2(dev,&props);
    printf("device: %s  api %u.%u.%u\n",props.properties.deviceName,
        VK_VERSION_MAJOR(props.properties.apiVersion),VK_VERSION_MINOR(props.properties.apiVersion),VK_VERSION_PATCH(props.properties.apiVersion));

    /* coopmat feature bits */
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cmprops = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmfeat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFloatControlsPropertiesKHR fc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR};
    cmfeat.pNext = &fc;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &cmfeat};
    vkGetPhysicalDeviceFeatures2(dev,&f2);
    printf("cooperativeMatrix feature: %d (stages=0x%x)\n",
        cmfeat.cooperativeMatrix, cmprops.cooperativeMatrixSupportedStages);
    printf("shaderSignedZeroInfNanPreserveFloat16/32: %d/%d  denorm f16: %d/%d/%d\n",
        fc.shaderSignedZeroInfNanPreserveFloat16,fc.shaderSignedZeroInfNanPreserveFloat32,
        fc.shaderDenormPreserveFloat16,fc.shaderDenormFlushToZeroFloat16,fc.shaderRoundingModeRTEFloat16);

    /* config enumeration */
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getcm =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(inst,"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if(!getcm){puts("no coopmat props fn");return 1;}
    uint32_t n=0; getcm(dev,&n,0);
    VkCooperativeMatrixPropertiesKHR* cm = calloc(n,sizeof*cm);
    for(uint32_t i=0;i<n;i++) cm[i].sType=VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    getcm(dev,&n,cm);
    printf("coopmat configs: %u\n%-28s %-28s %-6s %-12s %-8s %s\n",n,"A","B","C","Result","MxNxK","scope");
    for(uint32_t i=0;i<n;i++){
        char ms[24],ns[24],ks[24];
        snprintf(ms,24,"%s",ctype(cm[i].AType));
        snprintf(ns,24,"%s",ctype(cm[i].BType));
        snprintf(ks,24,"%s",ctype(cm[i].CType));
        printf("%-28s %-28s %-6s %-12s %ux%ux%u   %s\n",ms,ns,ks,ctype(cm[i].ResultType),
            cm[i].MSize,cm[i].NSize,cm[i].KSize,scope(cm[i].scope));
    }

    /* queues */
    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(dev,&nq,0);
    VkQueueFamilyProperties* q = calloc(nq,sizeof*q);
    vkGetPhysicalDeviceQueueFamilyProperties(dev,&nq,q);
    printf("\nqueue families: %u\n",nq);
    for(uint32_t i=0;i<nq;i++)
        printf("  fam%u count=%u flags=0x%x\n",i,q[i].queueCount,q[i].queueFlags);

    /* memory heaps */
    VkPhysicalDeviceMemoryProperties2 mp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    mp.pNext=&bud;
    vkGetPhysicalDeviceMemoryProperties2(dev,&mp);
    printf("heaps: %u\n",mp.memoryProperties.memoryHeapCount);
    for(uint32_t i=0;i<mp.memoryProperties.memoryHeapCount;i++)
        printf("  heap%u size=%.2fGiB budget=%.2fGiB flags=0x%x\n",i,
            mp.memoryProperties.memoryHeaps[i].size/1073741824.0,
            bud.heapBudget[i]/1073741824.0,mp.memoryProperties.memoryHeaps[i].flags);

    free(cm);free(q);free(devs);vkDestroyInstance(inst,0);
    return 0;
}
