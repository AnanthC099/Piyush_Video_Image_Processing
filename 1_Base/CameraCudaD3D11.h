#pragma once

// Windows / D3D / Media Foundation
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

// Standard C++/C
#include <atomic>
#include <vector>

// CUDA interop
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// Global variables
extern HWND                  g_hWnd;
extern ID3D11Device* g_pD3DDevice;
extern ID3D11DeviceContext* g_pD3DContext;
extern IDXGISwapChain* g_pSwapChain;
extern ID3D11RenderTargetView* g_pRTV;

extern ID3D11Texture2D* g_pCameraStagingTex;
extern ID3D11Texture2D* g_pCameraSharedTex;
extern ID3D11ShaderResourceView* g_pCameraSharedSRV;

extern ID3D11VertexShader* g_pVertexShader;
extern ID3D11PixelShader* g_pPixelShader;
extern ID3D11InputLayout* g_pInputLayout;

extern IMFSourceReader* g_pSourceReaderVideo;
extern UINT                  g_frameWidth;
extern UINT                  g_frameHeight;
extern LONG                  g_cameraStride;
extern UINT32                g_interlaceMode;

extern CRITICAL_SECTION      g_bufferLock;
extern std::vector<BYTE>     g_cpuBuffer;

extern HANDLE                g_hCaptureThread;
extern std::atomic<bool>     g_bRunning;

extern BOOL                  gbFullscreen;
extern DWORD                 dwStyle;
extern WINDOWPLACEMENT       wpPrev;

// CUDA interop resource
extern cudaGraphicsResource* g_pCudaGraphicsResource;

// Function prototypes
HRESULT InitCamera();
HRESULT InitD3D11(int suggestedWidth, int suggestedHeight);
HRESULT InitCUDA();
void    CleanupD3D11();
DWORD WINAPI CaptureThreadProc(LPVOID);
void    RenderFrame();
void    CreateSimpleShaders();
bool    SaveRGB32AsBMP(const char* filename, const BYTE* data, UINT width, UINT height);
void    ToggleFullscreen();
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
