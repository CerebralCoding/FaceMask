import Accelerate
import AVFoundation
import Foundation
import Vision

final class VideoAnalyzer: @unchecked Sendable {
    typealias StatusUpdateHandler = (AnalyzerStatus) -> Void
    typealias AnalyzedPercentageUpdateHandler = (Int) -> Void
    typealias WrittenPercentageUpdateHandler = (Int) -> Void
    typealias CancelHandler = () -> Bool

    var onStatusUpdate: StatusUpdateHandler?
    var onAnalyzedPercentageUpdate: AnalyzedPercentageUpdateHandler?
    var onWrittenPercentageUpdate: WrittenPercentageUpdateHandler?
    var onCancel: CancelHandler?

    var currentStatus: AnalyzerStatus = .idle

    var frameCount: Int = 0

    var writtenFrameCount: Int = 0

    var estimatedFrameCount: Int = 0

    var analysedPercentage: Int = 0

    var writtenPercentage: Int = 0

    var resultURL: URL? // Probably a really bad idea, but it solves some issues right now

    var isCancelled: Bool = false

    @Published var isExporting: Bool = false

    func analyze(inputURL: URL, userConfig: UserConfig, selectiveBlur _: Bool) async throws -> URL? {
        checkForCancellation()
        DispatchQueue.main.async {
            self.currentStatus = .analyzing
            self.onStatusUpdate?(self.currentStatus)
        }
        guard let videoReader = try? await VideoReader(inputURL: inputURL, userConfig: userConfig) else {
            throw AnalyzerError("Could not initialize video reader!")
        }

        guard let duration = videoReader.duration else {
            throw AnalyzerError("Could not calculate video duration!")
        }

        guard let fps = videoReader.fps else {
            throw AnalyzerError("Could not calculate frames per second!")
        }

        guard let transform = videoReader.transform else {
            throw AnalyzerError("Could not calculate video transform!")
        }

        guard let assetSize = videoReader.assetSize else {
            throw AnalyzerError("Could not calculate video size!")
        }

        let portraitOrientation = isPortrait(transform: transform, assetSize: assetSize)

        let rotationAngle = rotationAngleFromTransform(transform)

        var faceObservations = [FaceObservations]()

        var activeChains: [SimpleChain] = []

        var selectionCards = [SelectionCard]()

        var totalFrames: [Int] = [] // Easier to establish start and end with an array of all recorded frames

        var pixelBuffer: CVPixelBuffer!

        var frameWidth: Int!

        var frameHeight: Int!

        var prevHistogram: [[vImagePixelCount]] = Array(repeating: Array(repeating: 0, count: 256), count: 4)

        var prevPixelBuffer: CVPixelBuffer?

        let imageSequenceRequestHandler = VNSequenceRequestHandler()

        estimatedFrameCount = Int(CMTimeGetSeconds(duration) * Float64(fps))

        while true {
            checkForCancellation()
            guard let sampleBuffer = videoReader.nextFrame() else {
                break
            }
            detectFace(sampleBuffer: sampleBuffer)

            DispatchQueue.main.async {
                self.analysedPercentage = calculatePercentage(current: Double(self.frameCount), total: Double(self.estimatedFrameCount))
                self.onAnalyzedPercentageUpdate?(self.analysedPercentage)
            }
            frameCount += 1 // Counting frame here instead of inside detectFace function
            guard !isCancelled else {
                throw AnalyzerError("Operation was cancelled!")
            }
        }

        let videoData = videoReader.closeReader()
        guard activeChains.count > 0 else {
            throw AnalyzerError("No faces were found!")
        }

        estimatedFrameCount = frameCount // Hacky way to deal with percentage of written frames

        for chain in activeChains {
            selectionCards.append(chain.returnSelectionCard())
        }

        // Take the faceObservations and create an array with the frameNumbers and create a ObservationExport array

        var observationExport = [ObservationExport]()

        for observation in faceObservations {
            observationExport.append(ObservationExport(frameNumber: observation.frameNumber))
        }

        guard let videoWriter = await VideoWriter(observationExports: observationExport, videoData: videoData, selectionCards: selectionCards) else {
            throw AnalyzerError("Failed to initialize video writer")
        }

        let insertedObservationExport = insertSelectionCards(observationExports: observationExport, selectionCards: selectionCards)
        checkForCancellation()
        isExporting = true
        DispatchQueue.main.async {
            self.currentStatus = .exporting
            self.onStatusUpdate?(self.currentStatus)
        }

        try await writeVideoAndAudio(observationExports: insertedObservationExport, userConfig: userConfig, frameCount: frameCount)

        while true {
            if isExporting == false {
                break
            }
        }

        guard !isCancelled else {
            throw AnalyzerError("Operation was cancelled!")
        }

        func detectFace(sampleBuffer: CMSampleBuffer) {
            let currentTimeStamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
            frameWidth = CVPixelBufferGetWidth(pixelBuffer)
            frameHeight = CVPixelBufferGetHeight(pixelBuffer)

            var distance = 0.0
            var sceneChange = false

            let histogramsInSwift = calculateHistogram(pixelBuffer: pixelBuffer)

            // TODO: When a face i suddenly moving in one direction, but it's size (width, height) and orientation (pitch, roll, yaw) appears to be almost static, it's somewhat safe to assume it's due to camera movement. Add "elastic" trailing to compensate for non-blurred occurences.

            var homographicAlignmentObservations = [VNImageHomographicAlignmentObservation]()

            let homographicDetectionRequest = VNHomographicImageRegistrationRequest(targetedCVPixelBuffer: pixelBuffer, completionHandler: { (request: VNRequest, _: Error?) in
                if let results = request.results as? [VNImageHomographicAlignmentObservation] {
                    if !results.isEmpty {
                        homographicAlignmentObservations.append(contentsOf: results)
                    }
                } else {
                    return
                }
            })

            var translationAlignmentObservations = [VNImageTranslationAlignmentObservation]()

            let translationAlignmentRequest = VNTranslationalImageRegistrationRequest(targetedCVPixelBuffer: pixelBuffer, completionHandler: { (request: VNRequest, _: Error?) in
                if let results = request.results as? [VNImageTranslationAlignmentObservation] {
                    if !results.isEmpty {
                        translationAlignmentObservations.append(contentsOf: results)
                    }
                } else {
                    return
                }
            })

            // Compare the histograms here

            if frameCount == 0 {
                prevPixelBuffer = pixelBuffer // Just to ensure there is a zero point
                prevHistogram = histogramsInSwift
            }

            distance = chiSquaredDistanceNormalized(hist1: prevHistogram, hist2: histogramsInSwift)
            try? imageSequenceRequestHandler.perform([homographicDetectionRequest, translationAlignmentRequest], on: prevPixelBuffer!)

            prevHistogram = histogramsInSwift
            prevPixelBuffer = pixelBuffer

            if distance > 0.25 {
                sceneChange = true
            }

            if sceneChange {
                for chain in activeChains {
                    chain.isFinished = true
                }
            }

            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)

            let observedResults = analyzeFrame(imageRequestHandler: imageRequestHandler)

            let observations = observedResults.observations

            let humanObservations = observedResults.humanObservations

            let trifectas = collectFaceObservations(rawObservations: observations)

            var addedObservations: [(FaceTrifecta, VNHumanObservation?)] = []
            var updatedChains: Set<SimpleChain> = []
            let activeChainsSet = Set(activeChains.filter { !$0.isFinished })

            let matchedTrifectaHumanPairs = matchTrifectaToHuman(faceTrifectas: trifectas, humanObservations: humanObservations)
            let trifectaHumanPairs = matchedTrifectaHumanPairs.trifectaHumanPairs
            var unmatchedHumans = matchedTrifectaHumanPairs.unmatchedHumans

            for (trifecta, human) in trifectaHumanPairs {
                var bestMatchChain: SimpleChain?
                var highestSimilarity: Float = 0.0

                for chain in activeChainsSet {
                    if let lastObs = chain.observations.last?.observation {
                        let similarity = calculateFaceObservationSimilarity(face1: lastObs.observation, face2: trifecta.observation)
                        if similarity > highestSimilarity {
                            highestSimilarity = similarity
                            bestMatchChain = chain
                        }
                    }
                }

                if let bestChain = bestMatchChain, highestSimilarity > 0.1 {
                    if checkBestObservation(current: bestChain.bestObservation, new: trifecta) {
                        bestChain.updateBestObservation(newFaceImage: FaceEngine.grabHeadShot(observation: trifecta, pixelBuffer: pixelBuffer, frameWidth: frameWidth, frameHeight: frameHeight, rotationAngle: rotationAngle), newBestObservation: trifecta)
                    }

                    bestChain.addObservation(trifecta, frame: frameCount, humanObservation: human)
                    addedObservations.append((trifecta, human))
                    updatedChains.insert(bestChain)

                } else {
                    let newChain = SimpleChain(initialObservation: trifecta, faceImage: FaceEngine.grabHeadShot(observation: trifecta, pixelBuffer: pixelBuffer, frameWidth: frameWidth, frameHeight: frameHeight, rotationAngle: rotationAngle), startingFrame: frameCount, frameWidth: frameWidth, frameHeight: frameHeight, fps: fps, portraitOrientation: portraitOrientation, rotationAngle: rotationAngle, humanObservation: human)
                    activeChains.append(newChain)
                    addedObservations.append((trifecta, human))
                    updatedChains.insert(newChain)
                }
            }

            let chainsForPrediction = activeChainsSet.subtracting(updatedChains)

            for chain in chainsForPrediction {
                guard let predictedObservation = chain.predictedObservations.last else {
                    continue
                }

                var matchedHuman: VNHumanObservation?

                let unmatchedHumansArray = Array(unmatchedHumans)

                // There is a slight chance that humans get mismatched since a better matching prediction might be generated later

                if let closestHuman = findClosestHumanToFace(humanObservations: unmatchedHumansArray, faceTrifecta: predictedObservation.observation) {
                    if isTrifectaInsideHuman(trifecta: predictedObservation.observation, human: closestHuman),
                       isTrifectaSizeRelevantComparedToHuman(trifecta: predictedObservation.observation, human: closestHuman)
                    {
                        matchedHuman = closestHuman
                        unmatchedHumans.remove(closestHuman)
                    }
                }

                if checkBestObservation(current: chain.bestObservation, new: predictedObservation.observation) {
                    chain.updateBestObservation(newFaceImage: FaceEngine.grabHeadShot(observation: predictedObservation.observation, pixelBuffer: pixelBuffer, frameWidth: frameWidth, frameHeight: frameHeight, rotationAngle: rotationAngle), newBestObservation: predictedObservation.observation)
                }

                chain.addObservation(predictedObservation.observation, frame: predictedObservation.frameNumber, humanObservation: matchedHuman)
            }

            let observedCase = FaceObservations(timeStamp: currentTimeStamp, width: frameWidth, height: frameHeight, portraitOrientation: portraitOrientation, rotationAngle: rotationAngle, frameNumber: frameCount, histograms: histogramsInSwift, homographicAlignmentObservations: homographicAlignmentObservations, translationAlignmentObservations: translationAlignmentObservations, faceTrifectas: trifectas, humanObservations: humanObservations)

            faceObservations.append(observedCase)
            totalFrames.append(frameCount)
        }

        func checkBestObservation(current: FaceTrifecta, new: FaceTrifecta) -> Bool {
            if let currentBestQuality = current.quality {
                if let newQuality = new.quality {
                    newQuality > currentBestQuality ||
                        (newQuality == currentBestQuality && new.confidence > current.confidence)
                } else {
                    false
                }
            } else if new.quality != nil {
                true
            } else {
                new.confidence > current.confidence
            }
        }

        func normalizeHistogram(hist: [vImagePixelCount]) -> [Double] {
            let sum = Double(hist.reduce(0, +))
            return hist.map { Double($0) / sum }
        }

        func chiSquaredDistanceNormalized(hist1: [[vImagePixelCount]], hist2: [[vImagePixelCount]]) -> Double {
            var totalDistance = 0.0

            for i in 0 ..< 4 {
                let channelHist1 = normalizeHistogram(hist: hist1[i])
                let channelHist2 = normalizeHistogram(hist: hist2[i])

                for (a, b) in zip(channelHist1, channelHist2) {
                    let diff = a - b
                    let sum = a + b
                    if sum != 0 {
                        totalDistance += (diff * diff) / sum
                    }
                }
            }

            return totalDistance * 0.5
        }

        func calculateHistogram(pixelBuffer: CVPixelBuffer) -> [[vImagePixelCount]] {
            var inBuffer = vImage_Buffer()

            CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

            inBuffer.width = vImagePixelCount(frameWidth)
            inBuffer.height = vImagePixelCount(frameHeight)
            inBuffer.rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
            if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
                inBuffer.data = baseAddress
            }

            var histogramA = [vImagePixelCount](repeating: 0, count: 256)
            var histogramR = [vImagePixelCount](repeating: 0, count: 256)
            var histogramG = [vImagePixelCount](repeating: 0, count: 256)
            var histogramB = [vImagePixelCount](repeating: 0, count: 256)

            // Creating an array of pointers to the histograms
            var histograms = [UnsafeMutablePointer<vImagePixelCount>?]()
            histograms.append(&histogramA)
            histograms.append(&histogramR)
            histograms.append(&histogramG)
            histograms.append(&histogramB)

            // Making the array of pointers mutable
            histograms.withUnsafeMutableBufferPointer { ptr in
                if let baseAddress = ptr.baseAddress {
                    _ = vImageHistogramCalculation_ARGB8888(&inBuffer, baseAddress, vImage_Flags(kvImageNoFlags))
                }
            }

            // This should be added to the FaceObservation struct
            var histogramsInSwift: [[vImagePixelCount]] = []
            histogramsInSwift.append(Array(histogramA))
            histogramsInSwift.append(Array(histogramR))
            histogramsInSwift.append(Array(histogramG))
            histogramsInSwift.append(Array(histogramB))

            CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

            return histogramsInSwift
        }

        func isPortrait(transform: CGAffineTransform, assetSize: CGSize) -> Bool {
            // Create a rect with the asset's size
            let assetRect = CGRect(origin: .zero, size: assetSize)
            // Apply the transform to the rect to see the final dimensions
            let transformedRect = assetRect.applying(transform)

            // Use the helper function to get the rotation angle
            let degrees = rotationAngleFromTransform(transform)

            // Check if the rotation angle is approximately 90 or 270 degrees
            let isRotatedToPortrait = (degrees > 45 && degrees < 135) || (degrees > 225 && degrees < 315)

            // A video is considered portrait if its transformed height is greater than its transformed width
            let isSizePortrait = abs(transformedRect.height) > abs(transformedRect.width)

            return isRotatedToPortrait && isSizePortrait
        }

        func rotationAngleFromTransform(_ transform: CGAffineTransform) -> CGFloat {
            // Calculate the rotation angle from the transform
            let rotationAngle = atan2(transform.b, transform.a)

            // Normalize the angle to be within the range [-π, π]
            let normalizedAngle = atan2(sin(rotationAngle), cos(rotationAngle))

            // Convert the normalized angle to degrees from radians
            let degrees = normalizedAngle * 180 / .pi

            return degrees
        }

        func insertSelectionCards(observationExports: [ObservationExport], selectionCards: [SelectionCard]) -> [ObservationExport] {
            var modifiedFaceObservations = observationExports

            let cardsForBlurring = selectionCards.filter(\.blur)

            for card in cardsForBlurring {
                for observation in card.normalizedObservations {
                    let frameNumber = observation.frameNumber

                    if let index = modifiedFaceObservations.firstIndex(where: { $0.frameNumber == frameNumber }) {
                        var faceObs = modifiedFaceObservations[index]
                        faceObs.observation.append(observation)
                        faceObs.observation.sort(by: { $0.scaleFactor > $1.scaleFactor })
                        modifiedFaceObservations[index] = faceObs
                    }
                }
            }

            return modifiedFaceObservations
        }

        @Sendable func checkForCancellation() {
            if onCancel?() == true {
                isCancelled = true
            }
        }

        func writeVideoAndAudio(observationExports: [ObservationExport], userConfig: UserConfig, frameCount: Int) async throws {
            let videoQueue = DispatchQueue(label: "videoWriter", qos: .userInitiated)
            let audioQueue = DispatchQueue(label: "audioWriter", qos: .userInitiated)

            checkForCancellation()
            videoWriter.assetWriter.startWriting()
            videoWriter.assetWriter.startSession(atSourceTime: .zero)

            writeVideoOnQueue(videoQueue)
            if videoWriter.containsAudio == true {
                writeAudioOnQueue(audioQueue)
            }

            @Sendable func finishWritingProcess(for trackType: TrackType) {
                switch trackType {
                case .video:
                    if !videoWriter.writingVideoFinished {
                        videoWriter.writingVideoFinished = true
                        videoWriter.videoWriterInput.markAsFinished()
                    }
                case .audio:
                    if videoWriter.containsAudio, !videoWriter.writingAudioFinished {
                        videoWriter.writingAudioFinished = true
                        videoWriter.audioWriterInput?.markAsFinished()
                    }
                }

                // Check if all necessary writing processes are completed
                if videoWriter.writingVideoFinished, !videoWriter.containsAudio || videoWriter.writingAudioFinished {
                    videoWriter.assetWriter.finishWriting { [self] in
                        switch videoWriter.assetWriter.status {
                        case .completed:
                            resultURL = videoWriter.outputURL
                        default:
                            resultURL = nil // Failsafe
                        }
                        isExporting = false
                    }
                }
            }

            func writeVideoOnQueue(_ serialQueue: DispatchQueue) {
                guard videoWriter.videoReader.startReading() else {
                    finishWritingProcess(for: .video)
                    return
                }

                videoWriter.videoWriterInput.requestMediaDataWhenReady(on: serialQueue) {
                    while videoWriter.videoWriterInput.isReadyForMoreMediaData, videoWriter.writingVideoFinished == false {
                        autoreleasepool { () in
                            checkForCancellation()
                            guard self.isCancelled == false || videoWriter.isFailed else {
                                videoWriter.videoReader?.cancelReading()
                                finishWritingProcess(for: .video)
                                return
                            }

                            guard let sampleBuffer = videoWriter.videoReaderTrackOutput.copyNextSampleBuffer() else {
                                finishWritingProcess(for: .video)
                                return
                            }

                            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                                finishWritingProcess(for: .video)
                                return
                            }

                            let writtenFrameCount = self.writtenFrameCount

                            let currentObservation = observationExports[writtenFrameCount].observation

                            DispatchQueue.main.async {
                                let percentage = calculatePercentage(current: Double(writtenFrameCount), total: Double(frameCount))
                                self.onWrittenPercentageUpdate?(percentage)
                            }
                            self.writtenFrameCount += 1

                            guard let modifiedBuffer = try! FaceEngine.blurFace(sourcePixelBuffer: pixelBuffer, currentObservation: currentObservation, userConfig: userConfig) else {
                                videoWriter.videoReader?.cancelReading()
                                finishWritingProcess(for: .video)
                                return
                            }

                            let timeStamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

                            guard videoWriter.pixelBufferAdapter.append(modifiedBuffer, withPresentationTime: timeStamp) else {
                                videoWriter.videoReader?.cancelReading()
                                finishWritingProcess(for: .video)
                                return
                            }
                        }
                    }
                }
            }

            func writeAudioOnQueue(_ serialQueue: DispatchQueue) {
                guard videoWriter.audioReader.startReading() else {
                    finishWritingProcess(for: .audio)
                    return
                }
                videoWriter.audioWriterInput.requestMediaDataWhenReady(on: serialQueue) {
                    while videoWriter.audioWriterInput.isReadyForMoreMediaData, videoWriter.writingAudioFinished == false {
                        autoreleasepool { () in
                            checkForCancellation()
                            guard self.isCancelled == false || videoWriter.isFailed else {
                                videoWriter.audioReader?.cancelReading()
                                finishWritingProcess(for: .audio)
                                return
                            }

                            guard let sampleBuffer = videoWriter.audioReaderTrackOutput.copyNextSampleBuffer() else {
                                finishWritingProcess(for: .audio)
                                return
                            }

                            guard videoWriter.audioWriterInput.append(sampleBuffer) else {
                                videoWriter.audioReader?.cancelReading()
                                finishWritingProcess(for: .audio)
                                return
                            }
                        }
                    }
                }
            }
        }
        return resultURL
    }
}
