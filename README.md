# FaceMask

**Automatic face masking tool for video files, designed for privacy-first workflows on macOS.**

FaceMask is a Swift command-line utility that automatically detects and blurs faces in videos — fast, offline, and without any external dependencies. It’s built using the core components from my now-discontinued iOS app, Blufa.

> ⚠️ Much of the codebase dates back to 2022. It works, but will be gradually refactored and modernized.

## Description

FaceMask is designed to be lean, precise and very fast when doing automatic blurring of faces in videos. It relies only on built-in libraries in macOS (and iOS, ipadOS, visionOS for that matter), so no external computer vision or video handling frameworks is needed. Which means all blurring is done **completely offline!**

FaceMask supports all natively supported video formats on macOS including ProRes. FaceMask tries to keep the output as close as possible to the source with framerate, colorspace and orientation, while pruning as much metadata as possible including GPS data.

FaceMask uses a two cycle process (one for analyzing and one for writing), with an optional selective blurring feature based on the persons found during the analysis.

The actual blurring of the faces is done by grabbing the CVPixelBuffer of the current frame and passing it as a CIImage, applying a CIFilter to the targeted area before rendering the results back on top of the CVPixelBuffer, moving on to the next frame.
This also means it is entirely possible to create custom CIFilters via Metal.

### Building from source

If you want to compile the project yourself the latest version of Xcode is required (16.2 as of this writing) as well as [notcurses](https://github.com/dankamongmen/notcurses) for the TUI. 
Ensure it is installed via:

```
brew install notcurses
```

Clone the project and open it either in Xcode or compile it directly from the command line via:

```
swift build -c release
```

That's it.
