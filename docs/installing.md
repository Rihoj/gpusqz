# Installing

Pre-built packages are attached to each [GitHub
release](https://github.com/Rihoj/gpusqz/releases) (see [Releases and
versioning](releasing.md)). Every other build of the `build` workflow
(`.github/workflows/build.yml`) leaves them as workflow artifacts too.

| platform | installer | portable archive | backends |
|---|---|---|---|
| Ubuntu 22.04+, Debian 12+ | `gpusqz_<version>_amd64.deb` | – | CUDA, Vulkan |
| RHEL/Rocky/Alma 8+, Fedora | `gpusqz-<version>-1.x86_64.rpm` | – | CUDA, Vulkan |
| Windows 10/11 x64 | `gpusqz-<version>-win64.msi` | `gpusqz-<version>-win64.zip` | CUDA, Vulkan |
| macOS 11+ (Apple silicon and Intel) | `gpusqz-<version>-Darwin.pkg` | `gpusqz-<version>-Darwin.tar.gz` | Vulkan (MoltenVK) |

Every package has `gpusqz` and `gpusqz_refdec`; the Windows ones add the
MSVC runtime DLLs and the macOS ones `lib/libMoltenVK.dylib`.

## Installers and archives

- **Linux `.deb` / `.rpm`**: install with the distribution's package
  manager, e.g. `sudo apt install ./gpusqz_<version>_amd64.deb` or
  `sudo dnf install ./gpusqz-<version>-1.x86_64.rpm`.
- **Windows `.msi`** installs to `C:\Program Files\gpusqz\bin` and adds
  that to the system PATH (open a new terminal afterwards). Uninstall it
  from Settings → Apps. The installer is not code-signed yet, so
  SmartScreen asks first: *More info* → *Run anyway*. Silent install:
  `msiexec /i gpusqz-<version>-win64.msi /qn`.
- **macOS `.pkg`** installs to `/usr/local/gpusqz` and links `gpusqz` and
  `gpusqz_refdec` into `/usr/local/bin`. It is not signed or notarized
  yet, so Gatekeeper refuses a double-click: allow it under System
  Settings → Privacy & Security → *Open Anyway*, or install from a
  terminal with `sudo installer -pkg gpusqz-<version>-Darwin.pkg -target /`.
  To uninstall: `sudo rm -rf /usr/local/gpusqz /usr/local/bin/gpusqz
  /usr/local/bin/gpusqz_refdec && sudo pkgutil --forget io.github.rihoj.gpusqz`.
- **Archives** (`.zip`, `.tar.gz`) need no installation: unpack and run
  from `bin/`. Keep `bin/` and `lib/` together on macOS, and clear the
  download quarantine there first: `xattr -dr com.apple.quarantine <unpacked dir>`.

## What each GPU needs at run time

- **NVIDIA**: compute capability 7.0 (Volta) or newer and a driver that
  supports CUDA 12. The CUDA runtime is linked in, so no toolkit is
  needed. The packages carry native code for Volta through Blackwell plus
  PTX that newer GPUs compile at load time.
- **AMD** (e.g. Radeon RX 580): the driver's Vulkan support. On Linux
  that is Mesa's RADV (`mesa-vulkan-drivers` on Debian/Ubuntu,
  `mesa-vulkan-drivers` on Fedora/RHEL) with the Vulkan loader
  (`libvulkan1` / `vulkan-loader`). On Windows it is the AMD Adrenalin
  driver. (ROCm/HIP is not used: it dropped Polaris cards like the RX 580
  and doesn't exist for them on Windows.)
- **Apple silicon** (M1, M4 and later): nothing else to install. The macOS
  packages ship MoltenVK, which runs the Vulkan backend on Metal, in
  `lib/` next to `bin/`. A Homebrew `molten-vk` or the Vulkan SDK also
  work.
- **Intel**: the driver's Vulkan support (Mesa ANV on Linux).

Run `gpusqz devices` to see what gpusqz found. `gpusqz_refdec <in.gsz>
<out>` decompresses on the CPU anywhere, with no GPU at all.
