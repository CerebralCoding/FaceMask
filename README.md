# FaceMask

FaceMask is a Swift command line application for automatic blurring of faces in videos.

It's built using the core components from my now discontinued app Blufa.

*Please note I'm in the process of cleaning up the old codebase, and will gradually update this repo to a working state!*

## Description

FaceMask is designed to be lean, precise and very fast when doing automatic blurring of faces in videos. It relies only on built-in libraries in macOS (and iOS, ipadOS, visionOS for that matter), so no external computer vision or video handling frameworks is needed. Which means all blurring is done **completely offline!**

FaceMask supports all natively supported video formats on macOS including ProRes. FaceMask tries to keep the output as close as possible to the source with framerate, colorspace and orientation, while pruning as much metadata as possible including GPS data.

FaceMask uses a two cycle process (one for analyzing and one for writing), with an optional selective blurring feature based on the persons found during the analysis.

## Dependencies

[Notcurses](https://github.com/dankamongmen/notcurses) is used for the TUI. Ensure it is installed via:

```
brew install notcurses
```