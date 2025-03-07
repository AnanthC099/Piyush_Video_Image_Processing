#include "OGL.h"
#include "cudaKernels.cuh" // For ApplyCudaKernel(...), though it currently does nothing

#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// Link libraries
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

//--------------- Global Variables ---------------
HWND                  g_hWnd = nullptr;
ID3D11Device* g_pD3DDevice = nullptr;
ID3D11DeviceContext* g_pD3DContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_pRTV = nullptr;

ID3D11Texture2D* g_pCameraStagingTex = nullptr;
ID3D11Texture2D* g_pCameraSharedTex = nullptr;
ID3D11ShaderResourceView* g_pCameraSharedSRV = nullptr;

ID3D11VertexShader* g_pVertexShader = nullptr;
ID3D11PixelShader* g_pPixelShader = nullptr;
ID3D11InputLayout* g_pInputLayout = nullptr;

IMFSourceReader* g_pSourceReaderVideo = nullptr;
UINT                   g_frameWidth = 0;
UINT                   g_frameHeight = 0;
LONG                   g_cameraStride = 0;
UINT32                 g_interlaceMode = 0;

CRITICAL_SECTION       g_bufferLock = {};
std::vector<BYTE>      g_cpuBuffer;

HANDLE                 g_hCaptureThread = nullptr;
std::atomic<bool>      g_bRunning(false);

BOOL                   gbFullscreen = FALSE;
DWORD                  dwStyle = 0;
WINDOWPLACEMENT        wpPrev = { sizeof(WINDOWPLACEMENT) };

cudaGraphicsResource* g_pCudaGraphicsResource = nullptr;

// Macro
#define SAFE_RELEASE(x) if(x){ (x)->Release(); (x)=nullptr; }

// Forward-declarations of local helpers
static HRESULT ConfigureWindowAndD3D(int desiredW, int desiredH);
static void NoOpCUDA();  // We'll call a no-op for demonstration.

//--------------------------------------------------------------------------------------
// main()
//--------------------------------------------------------------------------------------
int main()
{
    timeBeginPeriod(1);

    AllocConsole();
    FILE* fDummy;
    freopen_s(&fDummy, "CONOUT$", "w", stdout);

    // Initialize COM + Media Foundation
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        printf("CoInitializeEx failed.\n");
        return -1;
    }
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        printf("MFStartup failed.\n");
        CoUninitialize();
        return -1;
    }

    // Initialize camera for RGB32 (which is usually BGRA in memory)
    if (FAILED(InitCamera()) || !g_pSourceReaderVideo) {
        printf("Could not open camera in RGB32.\n");
        return -1;
    }

    // If no valid negotiated resolution
    if (g_frameWidth == 0 || g_frameHeight == 0) {
        printf("Invalid camera resolution.\n");
        return -1;
    }

    // Setup window + D3D
    if (FAILED(ConfigureWindowAndD3D(g_frameWidth, g_frameHeight))) {
        printf("ConfigureWindowAndD3D failed.\n");
        return -1;
    }

    // Initialize CUDA (register the shared D3D texture)
    if (FAILED(InitCUDA())) {
        printf("InitCUDA failed.\n");
        return -1;
    }

    // Start capture thread
    g_bRunning = true;
    InitializeCriticalSection(&g_bufferLock);
    g_hCaptureThread = CreateThread(nullptr, 0, CaptureThreadProc, nullptr, 0, nullptr);
    if (!g_hCaptureThread) {
        printf("Failed to create capture thread!\n");
        g_bRunning = false;
    }

    // Main message loop
    MSG msg = {};
    while (true) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else {
            if (g_bRunning) {
                RenderFrame();
            }
        }
    }

    // Cleanup
    g_bRunning = false;
    if (g_hCaptureThread) {
        WaitForSingleObject(g_hCaptureThread, INFINITE);
        CloseHandle(g_hCaptureThread);
        g_hCaptureThread = nullptr;
    }
    DeleteCriticalSection(&g_bufferLock);

    SAFE_RELEASE(g_pSourceReaderVideo);
    CleanupD3D11();

    MFShutdown();
    CoUninitialize();

    timeEndPeriod(1);
    FreeConsole();
    return 0;
}

//--------------------------------------------------------------------------------------
// ConfigureWindowAndD3D
//--------------------------------------------------------------------------------------
static HRESULT ConfigureWindowAndD3D(int desiredW, int desiredH)
{
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);

    float aspect = (float)desiredW / (float)desiredH;

    // Scale down if bigger than screen
    if (desiredW > screenW) {
        desiredW = screenW;
        desiredH = (int)((float)screenW / aspect);
    }
    if (desiredH > screenH) {
        desiredH = screenH;
        desiredW = (int)((float)screenH * aspect);
    }

    RECT rc = { 0, 0, desiredW, desiredH };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    int finalW = rc.right - rc.left;
    int finalH = rc.bottom - rc.top;

    int posX = (screenW - finalW) / 2;
    int posY = (screenH - finalH) / 2;

    WNDCLASSA wc = {};
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = "CamPreviewClass_Centered";

    ATOM atomResult = RegisterClassA(&wc);
    if (!atomResult) {
        DWORD lastErr = GetLastError();
        printf("RegisterClass failed! GetLastError()=%lu\n", lastErr);
        return E_FAIL;
    }

    g_hWnd = CreateWindowA(
        wc.lpszClassName,
        "WMF + D3D11 + CUDA Interop (No Effects)",
        WS_OVERLAPPEDWINDOW,
        posX, posY,
        finalW, finalH,
        NULL, NULL,
        wc.hInstance,
        NULL);

    if (!g_hWnd) {
        DWORD lastErr = GetLastError();
        printf("CreateWindow failed! GetLastError()=%lu\n", lastErr);
        return E_FAIL;
    }

    ShowWindow(g_hWnd, SW_SHOW);
    UpdateWindow(g_hWnd);

    // Initialize D3D
    HRESULT hr = InitD3D11(desiredW, desiredH);
    if (FAILED(hr)) {
        printf("InitD3D11 failed! HRESULT=0x%08X\n", (unsigned)hr);
        return hr;
    }
    return S_OK;
}

//--------------------------------------------------------------------------------------
// InitCamera
//--------------------------------------------------------------------------------------
HRESULT InitCamera()
{
    IMFAttributes* pAttrEnum = nullptr;
    IMFActivate** ppDevices = nullptr;
    UINT32         count = 0;

    HRESULT hr = MFCreateAttributes(&pAttrEnum, 1);
    if (SUCCEEDED(hr)) {
        hr = pAttrEnum->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    }
    if (FAILED(hr)) {
        SAFE_RELEASE(pAttrEnum);
        return hr;
    }

    hr = MFEnumDeviceSources(pAttrEnum, &ppDevices, &count);
    SAFE_RELEASE(pAttrEnum);
    if (FAILED(hr) || count == 0) {
        if (ppDevices) CoTaskMemFree(ppDevices);
        printf("No camera devices found.\n");
        return E_FAIL;
    }

    printf("Video Capture Devices:\n");
    for (UINT32 i = 0; i < count; i++) {
        WCHAR* szName = nullptr;
        UINT32 cchName = 0;
        if (SUCCEEDED(ppDevices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &szName, &cchName))) {
            wprintf(L"  [%u] %s\n", i, szName);
            CoTaskMemFree(szName);
        }
        else {
            wprintf(L"  [%u] (Unknown)\n", i);
        }
    }

    printf("Select a video device index: ");
    UINT32 choice = 0;
    std::cin >> choice;
    if (choice >= count) {
        printf("Invalid choice.\n");
        for (UINT32 i = 0; i < count; i++) SAFE_RELEASE(ppDevices[i]);
        CoTaskMemFree(ppDevices);
        return E_FAIL;
    }

    IMFMediaSource* pSource = nullptr;
    hr = ppDevices[choice]->ActivateObject(IID_PPV_ARGS(&pSource));
    for (UINT32 i = 0; i < count; i++) SAFE_RELEASE(ppDevices[i]);
    CoTaskMemFree(ppDevices);
    if (FAILED(hr)) {
        printf("ActivateObject failed.\n");
        return hr;
    }

    // Create SourceReader
    IMFAttributes* pReaderAttr = nullptr;
    hr = MFCreateAttributes(&pReaderAttr, 1);
    if (SUCCEEDED(hr)) {
        hr = pReaderAttr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    }
    if (SUCCEEDED(hr)) {
        hr = MFCreateSourceReaderFromMediaSource(pSource, pReaderAttr, &g_pSourceReaderVideo);
    }
    SAFE_RELEASE(pReaderAttr);
    SAFE_RELEASE(pSource);
    if (FAILED(hr)) {
        printf("MFCreateSourceReaderFromMediaSource failed.\n");
        return hr;
    }

    // Request RGB32 (which usually is BGRA in memory)
    {
        IMFMediaType* pType = nullptr;
        hr = MFCreateMediaType(&pType);
        if (SUCCEEDED(hr)) hr = pType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        if (SUCCEEDED(hr)) hr = pType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        if (SUCCEEDED(hr)) {
            hr = g_pSourceReaderVideo->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, pType);
        }
        SAFE_RELEASE(pType);
        if (FAILED(hr)) {
            printf("Unable to set camera to RGB32.\n");
            return hr;
        }
    }

    // Get final format
    {
        IMFMediaType* pCurrent = nullptr;
        hr = g_pSourceReaderVideo->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &pCurrent);
        if (SUCCEEDED(hr) && pCurrent) {
            UINT32 w = 0, h = 0;
            MFGetAttributeSize(pCurrent, MF_MT_FRAME_SIZE, &w, &h);
            g_frameWidth = w;
            g_frameHeight = h;
            pCurrent->GetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32*)&g_cameraStride);
            pCurrent->GetUINT32(MF_MT_INTERLACE_MODE, &g_interlaceMode);
            pCurrent->Release();
        }
    }
    printf("Camera finalized at %ux%u, stride=%d, interlaceMode=%u\n",
        g_frameWidth, g_frameHeight, g_cameraStride, g_interlaceMode);

    return S_OK;
}

//--------------------------------------------------------------------------------------
// InitD3D11
//--------------------------------------------------------------------------------------
HRESULT InitD3D11(int width, int height)
{
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // <--- B8G8R8A8 to match camera's BGRA
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 1;
    sd.OutputWindow = g_hWnd;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT creationFlags = 0;
#ifdef _DEBUG
    creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };

    ID3D11Device* pTempDevice = nullptr;
    ID3D11DeviceContext* pTempContext = nullptr;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        creationFlags,
        levels,
        _countof(levels),
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &pTempDevice,
        nullptr,
        &pTempContext);
    if (FAILED(hr)) {
        printf("D3D11CreateDeviceAndSwapChain failed. HRESULT=0x%08X\n", (unsigned)hr);
        return hr;
    }

    // QI real device/context
    hr = pTempDevice->QueryInterface(__uuidof(ID3D11Device), (void**)&g_pD3DDevice);
    SAFE_RELEASE(pTempDevice);
    if (FAILED(hr)) {
        printf("QueryInterface(ID3D11Device) failed. HRESULT=0x%08X\n", (unsigned)hr);
        return hr;
    }

    hr = pTempContext->QueryInterface(__uuidof(ID3D11DeviceContext), (void**)&g_pD3DContext);
    SAFE_RELEASE(pTempContext);
    if (FAILED(hr)) {
        printf("QueryInterface(ID3D11DeviceContext) failed. HRESULT=0x%08X\n", (unsigned)hr);
        return hr;
    }

    // Backbuffer RT
    {
        ID3D11Texture2D* pBB = nullptr;
        hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pBB);
        if (SUCCEEDED(hr)) {
            hr = g_pD3DDevice->CreateRenderTargetView(pBB, nullptr, &g_pRTV);
            pBB->Release();
        }
        if (FAILED(hr)) {
            printf("CreateRenderTargetView failed. HRESULT=0x%08X\n", (unsigned)hr);
            return hr;
        }
    }

    // Create staging texture for CPU writes
    {
        D3D11_TEXTURE2D_DESC tdesc = {};
        tdesc.Width = g_frameWidth;
        tdesc.Height = g_frameHeight;
        tdesc.MipLevels = 1;
        tdesc.ArraySize = 1;
        tdesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // <--- B8G8R8A8
        tdesc.SampleDesc.Count = 1;
        tdesc.Usage = D3D11_USAGE_STAGING;
        tdesc.BindFlags = 0;
        tdesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        tdesc.MiscFlags = 0;

        hr = g_pD3DDevice->CreateTexture2D(&tdesc, nullptr, &g_pCameraStagingTex);
        if (FAILED(hr)) {
            printf("CreateTexture2D (staging) failed. HRESULT=0x%08X\n", (unsigned)hr);
            return hr;
        }
    }

    // Create shared texture for GPU + CUDA
    {
        D3D11_TEXTURE2D_DESC tdesc = {};
        tdesc.Width = g_frameWidth;
        tdesc.Height = g_frameHeight;
        tdesc.MipLevels = 1;
        tdesc.ArraySize = 1;
        tdesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // <--- B8G8R8A8
        tdesc.SampleDesc.Count = 1;
        tdesc.Usage = D3D11_USAGE_DEFAULT;
        tdesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        tdesc.CPUAccessFlags = 0;
        tdesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        hr = g_pD3DDevice->CreateTexture2D(&tdesc, nullptr, &g_pCameraSharedTex);
        if (FAILED(hr)) {
            printf("CreateTexture2D (shared) failed. HRESULT=0x%08X\n", (unsigned)hr);
            return hr;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // <--- B8G8R8A8
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Texture2D.MostDetailedMip = 0;

        hr = g_pD3DDevice->CreateShaderResourceView(g_pCameraSharedTex, &srvDesc, &g_pCameraSharedSRV);
        if (FAILED(hr)) {
            printf("CreateShaderResourceView failed. HRESULT=0x%08X\n", (unsigned)hr);
            return hr;
        }
    }

    // Viewport
    {
        D3D11_VIEWPORT vp = {};
        vp.Width = (float)width;
        vp.Height = (float)height;
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        vp.TopLeftX = 0.f;
        vp.TopLeftY = 0.f;
        g_pD3DContext->RSSetViewports(1, &vp);
    }

    // Create pass-through shaders
    CreateSimpleShaders();
    return S_OK;
}

//--------------------------------------------------------------------------------------
// InitCUDA
//--------------------------------------------------------------------------------------
HRESULT InitCUDA()
{
    // Force creation of a CUDA context
    cudaError_t cuErr = cudaFree(0);
    if (cuErr != cudaSuccess) {
        printf("CUDA init (cudaFree(0)) failed: %s\n", cudaGetErrorString(cuErr));
        return E_FAIL;
    }

    // Register resource with CUDA
    cuErr = cudaGraphicsD3D11RegisterResource(
        &g_pCudaGraphicsResource,
        g_pCameraSharedTex,
        cudaGraphicsRegisterFlagsNone
    );
    if (cuErr != cudaSuccess) {
        printf("cudaGraphicsD3D11RegisterResource failed: %s\n", cudaGetErrorString(cuErr));
        return E_FAIL;
    }
    printf("CUDA resource registered (no-op usage).\n");
    return S_OK;
}

//--------------------------------------------------------------------------------------
// CleanupD3D11
//--------------------------------------------------------------------------------------
void CleanupD3D11()
{
    if (g_pCudaGraphicsResource) {
        cudaGraphicsUnregisterResource(g_pCudaGraphicsResource);
        g_pCudaGraphicsResource = nullptr;
    }

    SAFE_RELEASE(g_pInputLayout);
    SAFE_RELEASE(g_pVertexShader);
    SAFE_RELEASE(g_pPixelShader);

    SAFE_RELEASE(g_pCameraSharedSRV);
    SAFE_RELEASE(g_pCameraSharedTex);
    SAFE_RELEASE(g_pCameraStagingTex);

    SAFE_RELEASE(g_pRTV);
    SAFE_RELEASE(g_pSwapChain);
    SAFE_RELEASE(g_pD3DContext);
    SAFE_RELEASE(g_pD3DDevice);
}

//--------------------------------------------------------------------------------------
// CaptureThreadProc
//--------------------------------------------------------------------------------------
DWORD WINAPI CaptureThreadProc(LPVOID)
{
    g_cpuBuffer.resize(g_frameWidth * g_frameHeight * 4, 0);
    bool savedFrame = false;

    while (g_bRunning) {
        IMFSample* pSample = nullptr;
        DWORD      streamFlags = 0;
        LONGLONG   llTimestamp = 0;

        HRESULT hr = g_pSourceReaderVideo->ReadSample(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,
            nullptr,
            &streamFlags,
            &llTimestamp,
            &pSample);

        if (FAILED(hr) || (streamFlags & MF_SOURCE_READERF_ENDOFSTREAM)) {
            if (pSample) pSample->Release();
            break;
        }

        if (pSample) {
            IMFMediaBuffer* pBuf = nullptr;
            if (SUCCEEDED(pSample->ConvertToContiguousBuffer(&pBuf)) && pBuf) {
                BYTE* pData = nullptr;
                DWORD maxLen = 0, curLen = 0;
                if (SUCCEEDED(pBuf->Lock(&pData, &maxLen, &curLen))) {
                    LONG strideIn = g_cameraStride;
                    LONG rowSize = (LONG)(g_frameWidth * 4);
                    if (strideIn == 0) strideIn = rowSize;

                    EnterCriticalSection(&g_bufferLock);
                    for (LONG row = 0; row < (LONG)g_frameHeight; ++row) {
                        LONG srcOff = row * strideIn;
                        LONG dstOff = row * rowSize;
                        if (srcOff + rowSize <= (LONG)curLen) {
                            memcpy(&g_cpuBuffer[dstOff], pData + srcOff, rowSize);
                        }
                        else {
                            break;
                        }
                    }
                    LeaveCriticalSection(&g_bufferLock);

                    if (!savedFrame) {
                        SaveRGB32AsBMP("debug_frame_noeffects.bmp",
                            g_cpuBuffer.data(),
                            g_frameWidth,
                            g_frameHeight);
                        savedFrame = true;
                    }

                    pBuf->Unlock();
                }
                pBuf->Release();
            }
            pSample->Release();
        }
    }
    return 0;
}

//--------------------------------------------------------------------------------------
// RenderFrame
//--------------------------------------------------------------------------------------
void RenderFrame()
{
    float clearColor[4] = { 0.f, 0.f, 0.f, 1.f };
    g_pD3DContext->ClearRenderTargetView(g_pRTV, clearColor);

    // Copy CPU -> staging
    if (!g_cpuBuffer.empty()) {
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        HRESULT hr = g_pD3DContext->Map(g_pCameraStagingTex, 0, D3D11_MAP_WRITE, 0, &mapped);
        if (SUCCEEDED(hr)) {
            EnterCriticalSection(&g_bufferLock);

            const UINT rowBytes = g_frameWidth * 4;
            const BYTE* srcBase = g_cpuBuffer.data();
            BYTE* dstBase = (BYTE*)mapped.pData;

            for (UINT row = 0; row < g_frameHeight; ++row) {
                memcpy(dstBase + row * mapped.RowPitch,
                    srcBase + row * rowBytes,
                    rowBytes);
            }
            LeaveCriticalSection(&g_bufferLock);

            g_pD3DContext->Unmap(g_pCameraStagingTex, 0);
        }
    }

    // staging -> shared
    g_pD3DContext->CopyResource(g_pCameraSharedTex, g_pCameraStagingTex);

    // NO-OP CUDA
    NoOpCUDA();

    // Draw
    g_pD3DContext->OMSetRenderTargets(1, &g_pRTV, nullptr);
    g_pD3DContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    g_pD3DContext->VSSetShader(g_pVertexShader, nullptr, 0);
    g_pD3DContext->PSSetShader(g_pPixelShader, nullptr, 0);
    g_pD3DContext->PSSetShaderResources(0, 1, &g_pCameraSharedSRV);

    g_pD3DContext->Draw(4, 0);
    g_pSwapChain->Present(1, 0);
}

//--------------------------------------------------------------------------------------
// NoOpCUDA: calls an empty kernel, does nothing
//--------------------------------------------------------------------------------------
static void NoOpCUDA()
{
    if (!g_pCudaGraphicsResource) return;

    // Map/unmap resource, no actual processing
    cudaGraphicsMapResources(1, &g_pCudaGraphicsResource, 0);
    cudaGraphicsUnmapResources(1, &g_pCudaGraphicsResource, 0);
}

//--------------------------------------------------------------------------------------
// CreateSimpleShaders: pass-through (no channel swap needed)
//--------------------------------------------------------------------------------------
void CreateSimpleShaders()
{
    // Vertex Shader: pass-through
    const char* vsSrc = R"(
    struct VS_OUT {
        float4 pos : SV_POSITION;
        float2 uv  : TEXCOORD0;
    };
    VS_OUT mainVS(uint vid : SV_VertexID)
    {
        float2 posArray[4] = {
            float2(-1,  1),
            float2( 1,  1),
            float2(-1, -1),
            float2( 1, -1)
        };
        float2 uvArray[4] = {
            float2(0,0),
            float2(1,0),
            float2(0,1),
            float2(1,1)
        };
        VS_OUT o;
        o.pos = float4(posArray[vid], 0, 1);
        o.uv  = uvArray[vid];
        return o;
    }
    )";

    // Pixel Shader: pass-through
    const char* psSrc = R"(
    Texture2D texCam : register(t0);
    SamplerState samLinear : register(s0);

    struct PS_IN {
        float4 pos : SV_POSITION;
        float2 uv  : TEXCOORD0;
    };

    float4 mainPS(PS_IN input) : SV_Target
    {
        // Sample exactly as is. Since the texture is B8G8R8A8 and
        // the camera data is BGRA, no channel swap needed.
        return texCam.Sample(samLinear, input.uv);
    }
    )";

    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;

    // Compile VS
    HRESULT hr = D3DCompile(
        vsSrc, strlen(vsSrc),
        nullptr, nullptr, nullptr,
        "mainVS", "vs_4_0", 0, 0,
        &vsBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            printf("Vertex Shader Error: %s\n", (char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
        return;
    }
    g_pD3DDevice->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &g_pVertexShader);
    g_pD3DDevice->CreateInputLayout(nullptr, 0, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &g_pInputLayout);
    if (errorBlob) errorBlob->Release();
    vsBlob->Release();

    // Compile PS
    hr = D3DCompile(
        psSrc, strlen(psSrc),
        nullptr, nullptr, nullptr,
        "mainPS", "ps_4_0", 0, 0,
        &psBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            printf("Pixel Shader Error: %s\n", (char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
        return;
    }
    g_pD3DDevice->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &g_pPixelShader);
    if (errorBlob) errorBlob->Release();
    psBlob->Release();
}

//--------------------------------------------------------------------------------------
// SaveRGB32AsBMP
//--------------------------------------------------------------------------------------
bool SaveRGB32AsBMP(const char* filename, const BYTE* data, UINT width, UINT height)
{
#pragma pack(push,1)
    struct BMPFILEHEADER {
        WORD  bfType;
        DWORD bfSize;
        WORD  bfReserved1;
        WORD  bfReserved2;
        DWORD bfOffBits;
    };
    struct BMPINFOHEADER {
        DWORD biSize;
        LONG  biWidth;
        LONG  biHeight;
        WORD  biPlanes;
        WORD  biBitCount;
        DWORD biCompression;
        DWORD biSizeImage;
        LONG  biXPelsPerMeter;
        LONG  biYPelsPerMeter;
        DWORD biClrUsed;
        DWORD biClrImportant;
    };
#pragma pack(pop)

    if (!data || width == 0 || height == 0) return false;

    const UINT bytesPerPixel = 4;
    const UINT rowBytes = width * bytesPerPixel;
    const UINT imageSize = rowBytes * height;

    BMPFILEHEADER fileHeader = {};
    BMPINFOHEADER infoHeader = {};

    fileHeader.bfType = 0x4D42; // 'BM'
    fileHeader.bfOffBits = sizeof(fileHeader) + sizeof(infoHeader);
    fileHeader.bfSize = fileHeader.bfOffBits + imageSize;

    infoHeader.biSize = sizeof(infoHeader);
    infoHeader.biWidth = (LONG)width;
    infoHeader.biHeight = (LONG)height; // bottom-up
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 32;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imageSize;

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        printf("Failed to open %s\n", filename);
        return false;
    }
    ofs.write((const char*)&fileHeader, sizeof(fileHeader));
    ofs.write((const char*)&infoHeader, sizeof(infoHeader));

    // bottom-up
    for (UINT row = 0; row < height; ++row) {
        const BYTE* rowPtr = data + (height - 1 - row) * rowBytes;
        ofs.write((const char*)rowPtr, rowBytes);
    }
    ofs.close();
    return true;
}

//--------------------------------------------------------------------------------------
// ToggleFullscreen
//--------------------------------------------------------------------------------------
void ToggleFullscreen(void)
{
    MONITORINFO mi = { sizeof(MONITORINFO) };
    if (!gbFullscreen) {
        dwStyle = GetWindowLong(g_hWnd, GWL_STYLE);
        if (dwStyle & WS_OVERLAPPEDWINDOW) {
            if (GetWindowPlacement(g_hWnd, &wpPrev) &&
                GetMonitorInfo(MonitorFromWindow(g_hWnd, MONITORINFOF_PRIMARY), &mi)) {
                SetWindowLong(g_hWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(g_hWnd, HWND_TOP,
                    mi.rcMonitor.left, mi.rcMonitor.top,
                    mi.rcMonitor.right - mi.rcMonitor.left,
                    mi.rcMonitor.bottom - mi.rcMonitor.top,
                    SWP_NOZORDER | SWP_FRAMECHANGED);
            }
        }
        ShowCursor(FALSE);
        gbFullscreen = TRUE;
    }
    else {
        SetWindowPlacement(g_hWnd, &wpPrev);
        SetWindowLong(g_hWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
        SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
        ShowCursor(TRUE);
        gbFullscreen = FALSE;
    }
}

//--------------------------------------------------------------------------------------
// WndProc
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CHAR:
        if (LOWORD(wParam) == 'f' || LOWORD(wParam) == 'F') {
            ToggleFullscreen();
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        break;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}
