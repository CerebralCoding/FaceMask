# FaceMask

**Automatic face masking tool for video files, designed for privacy-first workflows on macOS.**

FaceMask is a Swift command-line utility that automatically detects and blurs faces in videos — fast, offline, and without any external dependencies. It’s built using the core components from my now-discontinued iOS app, Blufa.

> ⚠️ Much of the codebase dates back to 2022. It works, but will be gradually refactored and modernized.

---

## 🔍 Features

- ⚡ Ultra-fast automatic face detection and masking
- 🛡️ All processing is **completely offline** — no cloud, no third-party frameworks
- 🎥 Supports all native macOS video formats (including ProRes)
- 🎯 Output preserves framerate, orientation, color space — but strips GPS and metadata

---

## 🛠 How It Works

FaceMask performs a two-stage pipeline:

1. **Analysis** — Scans all frames and detects all visible faces
2. **Writing** — Applies a configurable `CIFilter` (e.g., blur or pixelate) to targeted regions and encodes the output

The system uses native Apple APIs (`Accelerate`, `AVFoundation`, `Vision`, `CoreImage`) and processes each frame by converting it to a `CIImage`, blurring the detected regions, and rendering it back onto the original pixel buffer.

---

## 🔧 Building from Source

You'll need the latest version of Xcode (≥ 16.2). No external dependencies are required.

```bash
swift build -c release
```

That’s it.

> 🛠️ Interactive features will be introduced in a future version and may require additional libraries.

---

## 📦 Coming Soon

- Optional **selective blurring** based on face identity
- Metal based CIFilters

---

## ❤️ Privacy First

FaceMask is built with a simple goal: **keep your media private** while still enabling powerful, automated face masking. No data ever leaves your machine.

---

## 📄 License

FaceMask is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, modify, and distribute this software with attribution.

> See [`LICENSE`](./LICENSE) for full details.

---

### 💫 Support This Project

If you find FaceMask useful, consider [starring the repo](https://github.com/CerebralCoding/FaceMask) or [sponsoring its development](https://github.com/sponsors/CerebralCoding) to support privacy-first, open-source tooling.
