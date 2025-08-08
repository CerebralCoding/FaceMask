import Accelerate
import AVFoundation
import Vision

final class VideoWriter: @unchecked Sendable {
    var observationExports: [ObservationExport]

    var videoData: VideoData

    var selectionCards: [SelectionCard]

    var outputURL: URL!

    var writtenFrameCount: Int = 0

    var formatDescription: [String: Any]?

    var containsAudio: Bool = false // Need to use boolean check to handle missing audio

    var assetWriter: AVAssetWriter!

    var videoWriterInput: AVAssetWriterInput!
    var audioWriterInput: AVAssetWriterInput!

    var pixelBufferAdapter: AVAssetWriterInputPixelBufferAdaptor!

    var videoReader: AVAssetReader!
    var videoTrack: AVAssetTrack!
    var videoReaderTrackOutput: AVAssetReaderTrackOutput!

    var audioReader: AVAssetReader!
    var audioTrack: AVAssetTrack!
    var audioReaderTrackOutput: AVAssetReaderTrackOutput!

    var writingVideoFinished = false
    var writingAudioFinished = false

    let rgbOutputSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBALE,
        kCVPixelBufferMetalCompatibilityKey as String: true,
    ]

    var isFailed = false

    init?(observationExports: [ObservationExport], videoData: VideoData, selectionCards: [SelectionCard]) async {
        self.observationExports = observationExports
        self.videoData = videoData
        self.selectionCards = selectionCards

        try? await initializeWriter()
    }

    func initializeWriter() async throws {
        let inputURL = videoData.inputURL
        let fileFormat = videoData.fileFormat
        let sourceFormatHintVideo = videoData.sourceFormatHintVideo
        let sourceFormatHintAudio = videoData.sourceFormatHintAudio
        let videoCodecs = videoData.videoCodecs

        outputURL = AppEnvironment.shared.outputURL

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: fileFormat) else {
            throw AnalyzerError("Could not initialize video writer!")
        }

        assetWriter = writer

        let asset = AVURLAsset(url: inputURL)

        guard let validVideo = try! await asset.loadTracks(withMediaType: .video).first else {
            throw AnalyzerError("Could not load video track!")
        }
        videoTrack = validVideo

        do {
            videoReader = try AVAssetReader(asset: asset)
        } catch {
            throw AnalyzerError("Could not initialize video reader!")
        }

        videoReaderTrackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: rgbOutputSettings)

        guard videoReaderTrackOutput != nil else {
            throw AnalyzerError("Could not generate video output!")
        }

        videoReaderTrackOutput.alwaysCopiesSampleData = true

        guard videoReader.canAdd(videoReaderTrackOutput) else {
            throw AnalyzerError("Failed to read video output!")
        }

        videoReader.add(videoReaderTrackOutput)

        if assetWriter.canApply(outputSettings: videoCodecs, forMediaType: AVMediaType.video) {
            let videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoCodecs, sourceFormatHint: sourceFormatHintVideo)
            videoWriterInput.transform = try! await videoTrack.load(.preferredTransform) // Solves wrong orientation on iOS video at least

            videoWriterInput.expectsMediaDataInRealTime = false

            if assetWriter.canAdd(videoWriterInput) {
                assetWriter.add(videoWriterInput)
                self.videoWriterInput = videoWriterInput
                pixelBufferAdapter = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: self.videoWriterInput) // IT WORKS WHEN THIS PIXELVALUE IS SET!!!
            }

            try await checkAudio(asset: asset, sourceFormatHintAudio: sourceFormatHintAudio)
        }
    }

    private func checkAudio(asset: AVAsset, sourceFormatHintAudio: CMFormatDescription?) async throws {
        if sourceFormatHintAudio != nil {
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
                throw AnalyzerError("Could not generate audio output!")
            }

            audioReaderTrackOutput.alwaysCopiesSampleData = true

            guard audioReader.canAdd(audioReaderTrackOutput) else {
                throw AnalyzerError("Could not read audio output!")
            }

            audioReader.add(audioReaderTrackOutput)

            if assetWriter.canApply(outputSettings: formatDescription, forMediaType: AVMediaType.audio) {
                let audioWriterInput = AVAssetWriterInput(mediaType: .audio, outputSettings: formatDescription, sourceFormatHint: sourceFormatHintAudio)

                audioWriterInput.expectsMediaDataInRealTime = false

                if assetWriter.canAdd(audioWriterInput) {
                    assetWriter.add(audioWriterInput)
                    self.audioWriterInput = audioWriterInput
                    containsAudio = true
                }
            }
        }
    }
}
