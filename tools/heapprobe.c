#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
int main(void){
    VkInstance inst; VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"hp",1,"",0,VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&app,0,0,0,0};
    if(vkCreateInstance(&ici,0,&inst)!=VK_SUCCESS){puts("inst fail");return 1;}
    uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0);
    printf("devs=%u\n",n); if(!n)return 1;
    VkPhysicalDevice* d=malloc(n*sizeof*d); vkEnumeratePhysicalDevices(inst,&n,d);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(d[0],&mp);
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d[0],&p);
    printf("%s\n",p.deviceName);
    for(uint32_t h=0;h<mp.memoryHeapCount;h++)
        printf("heap%u size=%.2fGB flags=0x%x\n",h,mp.memoryHeaps[h].size/1073741824.0,mp.memoryHeaps[h].flags);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        printf("type%u heap=%u flags=0x%x\n",i,mp.memoryTypes[i].heapIndex,mp.memoryTypes[i].propertyFlags);
    return 0;
}
