import AVFoundation

class VideoReader {
    var inputURL: URL

    var userConfig: UserConfig

    // The asset containing video and optional audio
    var asset: AVAsset!

    var duration: CMTime!

    var fps: Float!

    var transform: CGAffineTransform!

    var assetSize: CGSize!

    var formatDescription: [String: Any]?

    // Video related reading mandatory
    var videoReader: AVAssetReader!
    var videoTrack: AVAssetTrack!
    var videoReaderTrackOutput: AVAssetReaderTrackOutput!

    // Audio related reading
    var audioReader: AVAssetReader!
    var audioTrack: AVAssetTrack!
    var audioReaderTrackOutput: AVAssetReaderTrackOutput!

    var containsAudio: Bool = false // Need to use boolean check to handle missing audio

    var fileFormat: AVFileType!

    var videoCodecs: [String: Any]!

    var spatialVideo: Bool = false

    var sourceFormatHintVideo: CMFormatDescription?

    var sourceFormatHintAudio: CMFormatDescription?

    var frameCount: Int = 0

    // Changed output settings
    let rgbOutputSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferMetalCompatibilityKey as String: true,
    ]

    init?(inputURL: URL, userConfig: UserConfig) async throws {
        self.inputURL = inputURL
        self.userConfig = userConfig
        asset = AVURLAsset(url: inputURL)

        let fileEnding = inputURL.pathExtension.lowercased()

        fileFormat = getFileFormat(fileEnding: fileEnding)

        try await checkAudio()

        guard let validVideo = try! await asset.loadTracks(withMediaType: .video).first else {
            throw AnalyzerError("No video track found!")
        }
        videoTrack = validVideo
        fps = try await videoTrack.load(.nominalFrameRate)
        duration = try await videoTrack.load(.timeRange).duration
        transform = try await videoTrack.load(.preferredTransform)
        assetSize = try await videoTrack.load(.naturalSize)
        let videoFormatDescription = try await videoTrack.load(.formatDescriptions)
        videoCodecs = exportCodecs(formatDescription: videoFormatDescription.first!)
        spatialVideo = isSpatial(formatDescription: videoFormatDescription.first!)

        // Ensuring that the video is large enough to be processed
        guard assetSize.width >= 256, assetSize.height >= 256 else {
            throw AnalyzerError("Video too small!")
        }

        do {
            videoReader = try AVAssetReader(asset: asset)
        } catch {
            throw AnalyzerError("Could not initialize video reader!")
        }

        // Using the modified output settings
        videoReaderTrackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: rgbOutputSettings)
        guard videoReaderTrackOutput != nil else {
            throw AnalyzerError("Could not generate valid video output!")
        }

        videoReaderTrackOutput.alwaysCopiesSampleData = true
        guard videoReader.canAdd(videoReaderTrackOutput) else {
            throw AnalyzerError("Failed to read video output!")
        }
        videoReader.add(videoReaderTrackOutput)
        videoReader.startReading()
    }

    func checkAudio() async throws {
        guard let validAudio = try! await asset.loadTracks(withMediaType: .audio).first else {
            containsAudio = false
            return
        }
        audioTrack = validAudio

        do {
            audioReader = try AVAssetReader(asset: asset)
        } catch {
            throw AnalyzerError("Could not initialize audio reader!")
        }

        audioReaderTrackOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: formatDescription)

        guard audioReaderTrackOutput != nil else {
            throw AnalyzerError("Could not generate valid audio output!")
        }

        audioReaderTrackOutput.alwaysCopiesSampleData = true

        guard audioReader.canAdd(audioReaderTrackOutput) else {
            throw AnalyzerError("Failed to read audio output!")
        }

        audioReader.add(audioReaderTrackOutput)
        containsAudio = true

        // Collect sourceFormatHint below

        audioReader.startReading()

        while sourceFormatHintAudio == nil {
            guard let sampleBuffer = audioReaderTrackOutput.copyNextSampleBuffer() else {
                break
            }
            sourceFormatHintAudio = CMSampleBufferGetFormatDescription(sampleBuffer)
            if sourceFormatHintAudio != nil {
                audioReader.cancelReading()
                break
            }
        }
    }

    // Think the CMSampleBufferRef should be returned here as well to write to the extension
    func nextFrame() -> CMSampleBuffer? {
        guard let sampleBuffer = videoReaderTrackOutput.copyNextSampleBuffer() else {
            return nil
        }

        // Every time a samplebuffer is fetched this will guarantee the sourceFormatHint is available at least by the very end
        if sourceFormatHintVideo == nil {
            sourceFormatHintVideo = CMSampleBufferGetFormatDescription(sampleBuffer)
        }

        frameCount += 1

        return sampleBuffer
    }

    func closeReader() -> VideoData {
        let videoData = VideoData(inputURL: inputURL, userConfig: userConfig, frameCount: frameCount, fileFormat: fileFormat, videoCodecs: videoCodecs, spatialVideo: spatialVideo, sourceFormatHintVideo: sourceFormatHintVideo!, sourceFormatHintAudio: sourceFormatHintAudio)

        videoReader.cancelReading()

        return videoData
    }

    private func getFileFormat(fileEnding: String) -> AVFileType {
        switch fileEnding {
        case "mov":
            AVFileType.mov
        case "mp4":
            AVFileType.mp4
        case "m4v":
            AVFileType.m4v
        default:
            AVFileType.mp4
        }
    }

    private func exportCodecs(formatDescription: CMFormatDescription) -> [String: Any] {
        let mediaSubType = CMFormatDescriptionGetMediaSubType(formatDescription)
        switch mediaSubType {
        case kCMVideoCodecType_H264:
            return [AVVideoCodecKey: AVVideoCodecType.h264]
        case kCMVideoCodecType_HEVC:
            return [AVVideoCodecKey: AVVideoCodecType.hevc]
        case kCMVideoCodecType_HEVCWithAlpha:
            return [AVVideoCodecKey: AVVideoCodecType.hevcWithAlpha]
        case kCMVideoCodecType_JPEG:
            return [AVVideoCodecKey: AVVideoCodecType.jpeg]
        case kCMVideoCodecType_AppleProRes422:
            return [AVVideoCodecKey: AVVideoCodecType.proRes422]
        case kCMVideoCodecType_AppleProRes422HQ:
            return [AVVideoCodecKey: AVVideoCodecType.proRes422HQ]
        case kCMVideoCodecType_AppleProRes422LT:
            return [AVVideoCodecKey: AVVideoCodecType.proRes422LT]
        case kCMVideoCodecType_AppleProRes422Proxy:
            return [AVVideoCodecKey: AVVideoCodecType.proRes422Proxy]
        case kCMVideoCodecType_AppleProRes4444:
            return [AVVideoCodecKey: AVVideoCodecType.proRes4444]
        case kCMVideoCodecType_AppleProRes4444XQ:
            return [AVVideoCodecKey: AVVideoCodecType(rawValue: "ap4x")]
        default:
            return [AVVideoCodecKey: AVVideoCodecType.hevc]
        }
    }

    private func isSpatial(formatDescription: CMFormatDescription) -> Bool {
        let hasLeftEye = (formatDescription.tagCollections ?? []).contains {
            $0.contains { $0 == .stereoView(.leftEye) }
        }
        let hasRightEye = (formatDescription.tagCollections ?? []).contains {
            $0.contains { $0 == .stereoView(.rightEye) }
        }
        return hasLeftEye && hasRightEye
    }
}
