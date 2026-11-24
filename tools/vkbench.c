/* vkbench.c — measure H2D expert-stream bandwidth on this box.
   Paths measured:
     (a) memcpy into ReBAR host-visible device-local buf (CPU store -> PCIe -> VRAM)
     (b) vkCmdCopyBuffer host-RAM -> VRAM on the dedicated transfer queue (SDMA)
   Compile: tools\build-vkbench.bat */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <vulkan/vulkan.h>

static double now_s(void){
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart/(double)f.QuadPart;
}
#define SZ (256u*1024u*1024u)   /* 256 MiB per direction */
#define ITERS 8

static uint32_t memtype(VkPhysicalDevice dev, uint32_t want){
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(dev,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if((mp.memoryTypes[i].propertyFlags&want)==want) return i;
    return UINT32_MAX;
}

int main(void){
    VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO,0,"vkbench",1,"",0,VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,0,0,&app,0,0,0,0};
    VkInstance inst;
    if(vkCreateInstance(&ici,0,&inst)!=VK_SUCCESS){puts("inst fail");return 1;}
    uint32_t ndev=0; vkEnumeratePhysicalDevices(inst,&ndev,0);
    if(!ndev){puts("no dev");return 1;}
    VkPhysicalDevice* devs=malloc(ndev*sizeof*devs);
    vkEnumeratePhysicalDevices(inst,&ndev,devs);
    VkPhysicalDevice pd=devs[0];
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd,&p);
    printf("device: %s\n",p.deviceName);

    /* dedicated transfer queue family */
    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0);
    VkQueueFamilyProperties* q=calloc(nq,sizeof*q);
    vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,q);
    uint32_t tfam=UINT32_MAX;
    for(uint32_t i=0;i<nq;i++)
        if((q[i].queueFlags&VK_QUEUE_TRANSFER_BIT)&&!(q[i].queueFlags&(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT))) tfam=i;
    if(tfam==UINT32_MAX){for(uint32_t i=0;i<nq;i++) if(q[i].queueFlags&VK_QUEUE_TRANSFER_BIT) tfam=i;}
    printf("transfer family: %u (count %u)\n",tfam,q[tfam].queueCount);

    float prio=1.f;
    VkDeviceQueueCreateInfo qi={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,0,0,tfam,1,&prio};
    const char* dext[]={VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME};
    VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,0,0,1,&qi,0,0,1,dext,0};
    VkDevice dev;
    if(vkCreateDevice(pd,&dci,0,&dev)!=VK_SUCCESS){puts("dev fail");return 1;}
    VkQueue tq; vkGetDeviceQueue(dev,tfam,0,&tq);

    uint32_t hostT=memtype(pd,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uint32_t barT =memtype(pd,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    uint32_t locT =memtype(pd,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    printf("memtypes host=%u bar=%u devlocal=%u\n",hostT,barT,locT);

    /* buffers */
    VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,0,0,SZ,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_SHARING_MODE_EXCLUSIVE,0,0};
    VkBuffer bHost,bBar; vkCreateBuffer(dev,&bi,0,&bHost); vkCreateBuffer(dev,&bi,0,&bBar);
    VkDeviceMemory mHost,mBar;
    VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,0,SZ,hostT};
    if(vkAllocateMemory(dev,&ai,0,&mHost)!=VK_SUCCESS){puts("host alloc fail");return 1;}
    ai.memoryTypeIndex=barT;
    if(vkAllocateMemory(dev,&ai,0,&mBar)!=VK_SUCCESS){puts("bar alloc fail");return 1;}
    vkBindBufferMemory(dev,bHost,mHost,0); vkBindBufferMemory(dev,bBar,mBar,0);
    void *ph,*pb;
    vkMapMemory(dev,mHost,0,SZ,0,&ph); vkMapMemory(dev,mBar,0,SZ,0,&pb);
    memset(ph,0xAB,SZ);

    /* (a) CPU memcpy into ReBAR VRAM */
    { double t0=now_s();
      for(int i=0;i<ITERS;i++){ memset(pb,i,SZ); }
      double dt=(now_s()-t0)/ITERS;
      printf("memcpy->VRAM (ReBAR, non-temporal stores): %.2f GB/s\n",SZ/dt/1e9); }

    /* write-combining is sensitive to pattern; also plain memcpy */
    { double t0=now_s();
      for(int i=0;i<ITERS;i++){ memcpy(pb,ph,SZ); }
      double dt=(now_s()-t0)/ITERS;
      printf("memcpy->VRAM (plain memcpy):               %.2f GB/s\n",SZ/dt/1e9); }

    /* (b) vkCmdCopyBuffer on transfer queue */
    VkCommandPoolCreateInfo pci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,0,0,tfam};
    VkCommandPool pool; vkCreateCommandPool(dev,&pci,0,&pool);
    VkCommandBufferAllocateInfo cai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,pool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev,&cai,&cb);
    VkCommandBufferBeginInfo cbi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,0};
    vkBeginCommandBuffer(cb,&cbi);
    VkBufferCopy rc={0,0,SZ};
    vkCmdCopyBuffer(cb,bHost,bBar,1,&rc);
    vkEndCommandBuffer(cb);
    VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,0,0};
    VkFence fence; vkCreateFence(dev,&fi,0,&fence);
    VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,0,0,0,0,1,&cb,0,0};
    /* warm */
    vkQueueSubmit(tq,1,&si,fence); vkWaitForFences(dev,1,&fence,1,~0ull); vkResetFences(dev,1,&fence);
    double best=0;
    for(int i=0;i<ITERS;i++){
        double t0=now_s();
        vkQueueSubmit(tq,1,&si,fence); vkWaitForFences(dev,1,&fence,1,~0ull); vkResetFences(dev,1,&fence);
        double dt=now_s()-t0, g=SZ/dt/1e9; if(g>best)best=g;
    }
    printf("vkCmdCopyBuffer RAM->VRAM (SDMA, best of %d): %.2f GB/s\n",ITERS,best);

    vkDestroyFence(dev,fence,0);vkDestroyCommandPool(dev,pool,0);
    vkDestroyBuffer(dev,bHost,0);vkDestroyBuffer(dev,bBar,0);
    vkFreeMemory(dev,mHost,0);vkFreeMemory(dev,mBar,0);
    vkDestroyDevice(dev,0);vkDestroyInstance(inst,0);
    free(q);free(devs);
    return 0;
}
