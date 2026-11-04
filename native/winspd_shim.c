#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <rpc.h>
#include <winspd/winspd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef DWORD (__cdecl *VRAMDISK_READ)(void *, UINT64, UINT32, void *, int);
typedef DWORD (__cdecl *VRAMDISK_WRITE)(void *, UINT64, UINT32, const void *, int);
typedef DWORD (__cdecl *VRAMDISK_FLUSH)(void *, UINT64, UINT32);
typedef DWORD (__cdecl *VRAMDISK_UNMAP)(void *, UINT64, UINT32);

typedef struct _VRAMDISK_CALLBACKS {
    VRAMDISK_READ Read;
    VRAMDISK_WRITE Write;
    VRAMDISK_FLUSH Flush;
    VRAMDISK_UNMAP Unmap;
} VRAMDISK_CALLBACKS;

typedef DWORD (__cdecl *SPD_CREATE)(PWSTR, const SPD_STORAGE_UNIT_PARAMS *, const SPD_STORAGE_UNIT_INTERFACE *, SPD_STORAGE_UNIT **);
typedef VOID (__cdecl *SPD_DELETE)(SPD_STORAGE_UNIT *);
typedef DWORD (__cdecl *SPD_START)(SPD_STORAGE_UNIT *, ULONG);
typedef VOID (__cdecl *SPD_WAIT)(SPD_STORAGE_UNIT *);
typedef VOID (__cdecl *SPD_SHUTDOWN)(SPD_STORAGE_UNIT *);

typedef struct _VRAMDISK_WINSPD {
    HMODULE Module;
    SPD_STORAGE_UNIT *Unit;
    VRAMDISK_CALLBACKS Callbacks;
    void *UserContext;
} VRAMDISK_WINSPD;

static void status_good(SPD_STORAGE_UNIT_STATUS *Status)
{
    memset(Status, 0, sizeof(*Status));
    Status->ScsiStatus = SCSISTAT_GOOD;
}

static void status_error(SPD_STORAGE_UNIT_STATUS *Status, UINT8 Asc)
{
    SpdStorageUnitStatusSetSense(Status, SCSI_SENSE_MEDIUM_ERROR, Asc, 0);
}

static BOOLEAN storage_read(SPD_STORAGE_UNIT *Unit, PVOID Buffer,
    UINT64 BlockAddress, UINT32 BlockCount, BOOLEAN FlushFlag,
    SPD_STORAGE_UNIT_STATUS *Status)
{
    VRAMDISK_WINSPD *Device = (VRAMDISK_WINSPD *)Unit->UserContext;
    status_good(Status);
    if (FlushFlag && ERROR_SUCCESS != Device->Callbacks.Flush(
        Device->UserContext, BlockAddress, BlockCount)) {
        status_error(Status, SCSI_ADSENSE_WRITE_ERROR);
        return TRUE;
    }
    if (ERROR_SUCCESS != Device->Callbacks.Read(Device->UserContext,
        BlockAddress, BlockCount, Buffer, 0))
        status_error(Status, SCSI_ADSENSE_UNRECOVERED_ERROR);
    return TRUE;
}

static BOOLEAN storage_write(SPD_STORAGE_UNIT *Unit, PVOID Buffer,
    UINT64 BlockAddress, UINT32 BlockCount, BOOLEAN FlushFlag,
    SPD_STORAGE_UNIT_STATUS *Status)
{
    VRAMDISK_WINSPD *Device = (VRAMDISK_WINSPD *)Unit->UserContext;
    status_good(Status);
    if (ERROR_SUCCESS != Device->Callbacks.Write(Device->UserContext,
        BlockAddress, BlockCount, Buffer, 0)) {
        status_error(Status, SCSI_ADSENSE_WRITE_ERROR);
        return TRUE;
    }
    if (FlushFlag && ERROR_SUCCESS != Device->Callbacks.Flush(
        Device->UserContext, BlockAddress, BlockCount))
        status_error(Status, SCSI_ADSENSE_WRITE_ERROR);
    return TRUE;
}

static BOOLEAN storage_flush(SPD_STORAGE_UNIT *Unit, UINT64 BlockAddress,
    UINT32 BlockCount, SPD_STORAGE_UNIT_STATUS *Status)
{
    VRAMDISK_WINSPD *Device = (VRAMDISK_WINSPD *)Unit->UserContext;
    status_good(Status);
    if (ERROR_SUCCESS != Device->Callbacks.Flush(Device->UserContext,
        BlockAddress, BlockCount))
        status_error(Status, SCSI_ADSENSE_WRITE_ERROR);
    return TRUE;
}

static BOOLEAN storage_unmap(SPD_STORAGE_UNIT *Unit,
    SPD_UNMAP_DESCRIPTOR Descriptors[], UINT32 Count,
    SPD_STORAGE_UNIT_STATUS *Status)
{
    VRAMDISK_WINSPD *Device = (VRAMDISK_WINSPD *)Unit->UserContext;
    status_good(Status);
    for (UINT32 Index = 0; Index < Count; Index++) {
        if (ERROR_SUCCESS != Device->Callbacks.Unmap(Device->UserContext,
            Descriptors[Index].BlockAddress, Descriptors[Index].BlockCount)) {
            status_error(Status, SCSI_ADSENSE_WRITE_ERROR);
            break;
        }
    }
    return TRUE;
}

static const SPD_STORAGE_UNIT_INTERFACE StorageInterface = {
    storage_read,
    storage_write,
    storage_flush,
    storage_unmap,
    { 0 }
};

static FARPROC get_proc(HMODULE Module, const char *Name)
{
    return GetProcAddress(Module, Name);
}

__declspec(dllexport) DWORD __cdecl vramdisk_winspd_start(
    UINT64 CapacityBytes, UINT32 SectorSize, PWSTR DeviceName,
    const VRAMDISK_CALLBACKS *Callbacks, void *UserContext, void **PHandle)
{
    if (0 == PHandle || 0 == Callbacks || 0 == Callbacks->Read ||
        0 == Callbacks->Write || 0 == Callbacks->Flush ||
        0 == Callbacks->Unmap || 0 == CapacityBytes || 0 == SectorSize ||
        0 != CapacityBytes % SectorSize)
        return ERROR_INVALID_PARAMETER;
    *PHandle = 0;

    HMODULE Module = LoadLibraryW(L"winspd-x64.dll");
    if (0 == Module)
        return GetLastError();

    SPD_CREATE Create = (SPD_CREATE)get_proc(Module, "SpdStorageUnitCreate");
    SPD_DELETE Delete = (SPD_DELETE)get_proc(Module, "SpdStorageUnitDelete");
    SPD_START Start = (SPD_START)get_proc(Module, "SpdStorageUnitStartDispatcher");
    SPD_WAIT Wait = (SPD_WAIT)get_proc(Module, "SpdStorageUnitWaitDispatcher");
    SPD_SHUTDOWN Shutdown = (SPD_SHUTDOWN)get_proc(Module, "SpdStorageUnitShutdown");
    if (!Create || !Delete || !Start || !Wait || !Shutdown) {
        FreeLibrary(Module);
        return ERROR_PROC_NOT_FOUND;
    }

    VRAMDISK_WINSPD *Device = (VRAMDISK_WINSPD *)HeapAlloc(
        GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*Device));
    if (0 == Device) {
        FreeLibrary(Module);
        return ERROR_NOT_ENOUGH_MEMORY;
    }
    Device->Module = Module;
    Device->Callbacks = *Callbacks;
    Device->UserContext = UserContext;

    SPD_STORAGE_UNIT_PARAMS Params;
    memset(&Params, 0, sizeof(Params));
    RPC_STATUS UuidStatus = UuidCreate(&Params.Guid);
    if (RPC_S_OK != UuidStatus && RPC_S_UUID_LOCAL_ONLY != UuidStatus) {
        HeapFree(GetProcessHeap(), 0, Device);
        FreeLibrary(Module);
        return ERROR_GEN_FAILURE;
    }
    Params.BlockCount = CapacityBytes / SectorSize;
    Params.BlockLength = SectorSize;
    memcpy(Params.ProductId, "VRAMDISK", sizeof("VRAMDISK") - 1);
    memcpy(Params.ProductRevisionLevel, "1.0", sizeof("1.0") - 1);
    Params.WriteProtected = FALSE;
    Params.CacheSupported = FALSE;
    Params.UnmapSupported = TRUE;
    Params.MaxTransferLength = 1024 * 1024;

    DWORD Error = Create(DeviceName, &Params, &StorageInterface, &Device->Unit);
    if (ERROR_SUCCESS == Error) {
        Device->Unit->UserContext = Device;
        Error = Start(Device->Unit, 0);
    }
    if (ERROR_SUCCESS != Error) {
        if (Device->Unit)
            Delete(Device->Unit);
        HeapFree(GetProcessHeap(), 0, Device);
        FreeLibrary(Module);
        return Error;
    }

    *PHandle = Device;
    return ERROR_SUCCESS;
}

__declspec(dllexport) void __cdecl vramdisk_winspd_stop(void *Handle)
{
    VRAMDISK_WINSPD *Device = (VRAMDISK_WINSPD *)Handle;
    if (0 == Device)
        return;

    SPD_SHUTDOWN Shutdown = (SPD_SHUTDOWN)get_proc(Device->Module, "SpdStorageUnitShutdown");
    SPD_WAIT Wait = (SPD_WAIT)get_proc(Device->Module, "SpdStorageUnitWaitDispatcher");
    SPD_DELETE Delete = (SPD_DELETE)get_proc(Device->Module, "SpdStorageUnitDelete");
    if (Shutdown && Wait && Delete) {
        Shutdown(Device->Unit);
        Wait(Device->Unit);
        Delete(Device->Unit);
    }
    FreeLibrary(Device->Module);
    HeapFree(GetProcessHeap(), 0, Device);
}
