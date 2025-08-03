import Accelerate
import AVFoundation
import CoreImage
import Foundation
import Vision

/// The FaceEngine class is a static class containing the relevant calculations for sufficient blurring of faces
/// It works by comparing the total frame size, with the frame size of the face, calculating the base pixel size needed to obstruct the face
/// A scale factor should also make sure the pixels grow/shrink appropiately during the video

enum FaceEngine {
    nonisolated(unsafe) static let sharedCIContext: CIContext = .init(options: [CIContextOption.useSoftwareRenderer: false]) // Used the unsafe modifier here because the CIContext is not thread safe

    static func blurFace(sourcePixelBuffer: CVPixelBuffer, currentObservation: [NormalizedObservation], userConfig: UserConfig) throws -> CVPixelBuffer? {
        let lockFlags = CVPixelBufferLockFlags(rawValue: 0)

        let workingCopy = sourcePixelBuffer

        guard CVPixelBufferLockBaseAddress(workingCopy, lockFlags) == kCVReturnSuccess else {
            return nil
        }

        guard currentObservation.count != 0 else {
            CVPixelBufferUnlockBaseAddress(workingCopy, lockFlags)
            return workingCopy
        }
        // Sort the observations by size (width and height) so that the smallest faces are blurred first

        let sortedObservations = currentObservation.sorted { $0.observation.size.width * $0.observation.size.height < $1.observation.size.width * $1.observation.size.height }

        let rawFrame = CIImage(cvPixelBuffer: workingCopy)

        var outputImage = rawFrame

        for observation in sortedObservations {
            let circularShape = userConfig.circularShape
            let croppedFace = rawFrame.cropped(to: observation.observation)
            var blurredFace = try blurPattern(inputImage: croppedFace, scaleFactor: observation.scaleFactor, userConfig: userConfig).cropped(to: observation.observation) // Stupid workaround for custom filters

            if circularShape {
                let mask = try createOvalMask(observation: observation.observation)
                blurredFace = blurredFace.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: outputImage, kCIInputMaskImageKey: mask])
            }
            outputImage = blurredFace.composited(over: outputImage)
        }

        sharedCIContext.render(outputImage, to: workingCopy)

        CVPixelBufferUnlockBaseAddress(workingCopy, lockFlags)

        return workingCopy
    }

    static func blurPattern(inputImage: CIImage, scaleFactor: Int, userConfig: UserConfig) throws -> CIImage {
        let blur = userConfig.blurPattern
        let hexColor = userConfig.hexColor

        switch blur {
        case .pixelated:
            let pixelate = Pixelate()
            pixelate.inputImage = inputImage
            pixelate.scaleFactor = scaleFactor
            guard let blurredFace = pixelate.outputImage() else {
                throw AnalyzerError("Could not blur face!")
            }
            return blurredFace
        case .crystalized:
            let crystalize = Crystalize()
            crystalize.inputImage = inputImage
            crystalize.scaleFactor = scaleFactor
            guard let blurredFace = crystalize.outputImage() else {
                throw AnalyzerError("Could not blur face!")
            }
            return blurredFace
        case .honeycombed:
            let hexagonal = Hexagonal()
            hexagonal.inputImage = inputImage
            hexagonal.scaleFactor = scaleFactor
            guard let blurredFace = hexagonal.outputImage() else {
                throw AnalyzerError("Could not blur face!")
            }
            return blurredFace
        case .gaussian:
            let gaussian = Gaussian()
            gaussian.inputImage = inputImage
            gaussian.scaleFactor = scaleFactor
            guard let blurredFace = gaussian.outputImage() else {
                throw AnalyzerError("Could not blur face!")
            }
            return blurredFace
        case .colorblock:
            let colorBlock = Colorblock()
            colorBlock.inputImage = inputImage
            colorBlock.inputColor = CIColor(color: colorFromHex(hex: hexColor))
            guard let blurredFace = colorBlock.outputImage() else {
                throw AnalyzerError("Could not blur face!")
            }
            return blurredFace
        }
    }

    static func createOvalMask(observation: CGRect) throws -> CIImage {
        let center = CIVector(x: observation.midX, y: observation.midY)
        var size = observation.size
        if size.width >= size.height {
            size.height = size.width + 1
        }

        guard let context = CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            throw AnalyzerError("Could not create context for oval mask!")
        }

        context.setFillColor(gray: 1.0, alpha: 1.0)
        context.fillEllipse(in: CGRect(origin: .zero, size: size))

        guard let cgMask = context.makeImage() else {
            throw AnalyzerError("Could not create CGImage for oval mask!")
        }

        return CIImage(cgImage: cgMask).transformed(by: CGAffineTransform(translationX: center.x - size.width / 2, y: center.y - size.height / 2)).cropped(to: observation)
    }

    static func grabHeadShot(observation: FaceTrifecta, pixelBuffer: CVPixelBuffer, frameWidth: Int, frameHeight: Int, rotationAngle: CGFloat) -> URL {
        let uuid = UUID()
        let outputURL = AppEnvironment.shared.storageDirectory.appendingPathComponent(uuid.uuidString + ".jpg")

        // Calculate the bounding box of the face in the image's coordinate space
        let normalizedBoundingBox = VNImageRectForNormalizedRect(observation.observation.boundingBox, Int(frameWidth), Int(frameHeight))

        autoreleasepool {
            // Crop the face out of the full image
            var faceImageFull = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: normalizedBoundingBox)

            // TODO: Move the rotation and scaling inside the return selectionCard function instead of here

            // If there is a rotation, apply an affine transform to correct the orientation
            if rotationAngle != 0 {
                // Calculate the rotation needed to make the image upright
                // Convert to radians and adjust the direction of rotation
                let radians = -rotationAngle * (CGFloat.pi / 180)
                faceImageFull = faceImageFull.oriented(forExifOrientation: Int32(rotationAngle))

                // Create a new transform with the calculated rotation
                let rotation = CGAffineTransform(rotationAngle: radians)
                // Calculate the offset needed to keep the image centered
                let widthOffset = faceImageFull.extent.width / 2
                let heightOffset = faceImageFull.extent.height / 2
                let translation = CGAffineTransform(translationX: widthOffset, y: heightOffset)
                let transform = translation.inverted().concatenating(rotation).concatenating(translation)

                // Apply the transform to the image
                faceImageFull = faceImageFull.transformed(by: transform)
            }

            // Scale the image to a max dimension of 300 pixels if needed
            let maxDimension = max(faceImageFull.extent.width, faceImageFull.extent.height)
            let scaleFactor = 300 / maxDimension
            let scaleTransform = CGAffineTransform(scaleX: scaleFactor, y: scaleFactor)
            faceImageFull = faceImageFull.transformed(by: scaleTransform)

            // Save the image to a file

            try? sharedCIContext.writeJPEGRepresentation(of: faceImageFull, to: outputURL, colorSpace: CGColorSpaceCreateDeviceRGB())
        }

        return outputURL
    }

    // This can be useful for comparing scenes later on
    static func saveFrame(pixelBuffer: CVPixelBuffer) -> URL {
        let uuid = UUID()
        let outputURL = AppEnvironment.shared.storageDirectory.appendingPathComponent(uuid.uuidString + ".jpg")

        autoreleasepool {
            let videoFrame = CIImage(cvPixelBuffer: pixelBuffer)

            try? sharedCIContext.writeJPEGRepresentation(of: videoFrame, to: outputURL, colorSpace: CGColorSpaceCreateDeviceRGB())
        }

        return outputURL
    }
}

// MARK: Structs

// TODO: Refactor this to exclude repetitions
struct VideoData {
    var inputURL: URL
    var userConfig: UserConfig
    var frameCount: Int
    var fileFormat: AVFileType
    var videoCodecs: [String: Any]
    var spatialVideo: Bool
    var sourceFormatHintVideo: CMFormatDescription
    var sourceFormatHintAudio: CMFormatDescription?

    init(inputURL: URL, userConfig: UserConfig, frameCount: Int, fileFormat: AVFileType, videoCodecs: [String: Any], spatialVideo: Bool, sourceFormatHintVideo: CMFormatDescription, sourceFormatHintAudio: CMFormatDescription? = nil) {
        self.inputURL = inputURL
        self.userConfig = userConfig
        self.frameCount = frameCount
        self.fileFormat = fileFormat
        self.videoCodecs = videoCodecs
        self.spatialVideo = spatialVideo
        self.sourceFormatHintVideo = sourceFormatHintVideo
        self.sourceFormatHintAudio = sourceFormatHintAudio
    }
}

// TODO: This struct needs to keep track of different scenes by capturing the first registered and last registered frame to disk, along with all the face observations in between.
struct VideoScene {}

struct FaceObservations {
    var timeStamp: CMTime
    var width: Int
    var height: Int
    var portraitOrientation: Bool
    var rotationAngle: CGFloat
    var frameNumber: Int
    var histograms: [[vImagePixelCount]]
    var homographicAlignmentObservations: [VNImageHomographicAlignmentObservation]
    var translationAlignmentObservations: [VNImageTranslationAlignmentObservation]
    var faceTrifectas: [FaceTrifecta]
    var humanObservations: [VNHumanObservation]
    var normalizedObservations = [NormalizedObservation]()

    init(timeStamp: CMTime, width: Int, height: Int, portraitOrientation: Bool, rotationAngle: CGFloat, frameNumber: Int, histograms: [[vImagePixelCount]], homographicAlignmentObservations: [VNImageHomographicAlignmentObservation], translationAlignmentObservations: [VNImageTranslationAlignmentObservation], faceTrifectas: [FaceTrifecta], humanObservations: [VNHumanObservation]) {
        self.timeStamp = timeStamp
        self.width = width
        self.height = height
        self.portraitOrientation = portraitOrientation
        self.rotationAngle = rotationAngle
        self.frameNumber = frameNumber
        self.histograms = histograms
        self.homographicAlignmentObservations = homographicAlignmentObservations
        self.translationAlignmentObservations = translationAlignmentObservations
        self.faceTrifectas = faceTrifectas
        self.humanObservations = humanObservations
    }
}

struct PredictedObservations {
    var observation: FaceTrifecta
    var frameNumber: Int

    init(observation: FaceTrifecta, frameNumber: Int) {
        self.observation = observation
        self.frameNumber = frameNumber
    }
}

struct NormalizedObservation: Sendable {
    var frameNumber: Int
    var observation: CGRect
    var scaleFactor: Int

    init(frameNumber: Int, observation: CGRect, scaleFactor: Int) {
        self.frameNumber = frameNumber
        self.observation = observation
        self.scaleFactor = scaleFactor
    }
}

struct ObservationExport: Sendable {
    var frameNumber: Int
    var observation = [NormalizedObservation]()

    init(frameNumber: Int) {
        self.frameNumber = frameNumber
    }
}

struct UserConfig {
    var blurPattern: BlurPattern
    var circularShape: Bool
    var hexColor: String

    init(blurPattern: BlurPattern, circularShape: Bool, hexColor: String) {
        self.blurPattern = blurPattern
        self.circularShape = circularShape
        self.hexColor = hexColor
    }
}

struct AnalyzerError: Error, CustomDebugStringConvertible {
    var debugDescription: String
    init(_ debugDescription: String) { self.debugDescription = debugDescription }
}

struct SelectionCard: Sendable {
    let uuid: UUID
    let image: CGImage
    let normalizedObservations: [NormalizedObservation]
    var blur = true

    init(uuid: UUID, image: CGImage, normalizedObservations: [NormalizedObservation]) {
        self.uuid = uuid
        self.image = image
        self.normalizedObservations = normalizedObservations
    }
}

// TODO: Unscented Kalman Filter (UKF) and Rauch-Tung-Striebel Smoother (RTS)

struct Matrix {
    var data: [[Double]]
    var rows: Int { data.count }
    var cols: Int { data[0].count }

    // For multiplying a Matrix by a Double
    static func * (left: Matrix, right: Double) -> Matrix {
        var resultData: [[Double]] = []
        for row in left.data {
            let newRow = row.map { $0 * right }
            resultData.append(newRow)
        }
        return Matrix(data: resultData)
    }

    // For multiplying a Matrix by another Matrix
    static func * (lhs: Matrix, rhs: Matrix) -> Matrix? {
        guard lhs.cols == rhs.rows else { return nil }

        var result = Matrix(data: Array(repeating: Array(repeating: 0.0, count: rhs.cols), count: lhs.rows))
        for i in 0 ..< lhs.rows {
            for j in 0 ..< rhs.cols {
                for k in 0 ..< lhs.cols {
                    result.data[i][j] += lhs.data[i][k] * rhs.data[k][j]
                }
            }
        }
        return result
    }

    static func + (lhs: Matrix, rhs: Matrix) -> Matrix? {
        guard lhs.rows == rhs.rows, lhs.cols == rhs.cols else { return nil }

        var result = lhs
        for i in 0 ..< lhs.rows {
            for j in 0 ..< lhs.cols {
                result.data[i][j] += rhs.data[i][j]
            }
        }
        return result
    }

    static func - (lhs: Matrix, rhs: Matrix) -> Matrix? {
        guard lhs.rows == rhs.rows, lhs.cols == rhs.cols else { return nil }

        var result = lhs
        for i in 0 ..< lhs.rows {
            for j in 0 ..< lhs.cols {
                result.data[i][j] -= rhs.data[i][j]
            }
        }
        return result
    }

    static func identity(size: Int) -> Matrix {
        var data = Array(repeating: Array(repeating: 0.0, count: size), count: size)
        for i in 0 ..< size {
            data[i][i] = 1.0
        }
        return Matrix(data: data)
    }

    static func fromColumns(_ columns: [[Double]]) -> Matrix? {
        guard !columns.isEmpty else { return nil }

        let rowCount = columns[0].count
        for column in columns {
            if column.count != rowCount {
                return nil // Columns must have the same number of rows
            }
        }

        var data: [[Double]] = Array(repeating: Array(repeating: 0.0, count: columns.count), count: rowCount)

        for colIndex in 0 ..< columns.count {
            for rowIndex in 0 ..< rowCount {
                data[rowIndex][colIndex] = columns[colIndex][rowIndex]
            }
        }

        return Matrix(data: data)
    }

    // Method to extract a column from the matrix and return it as a new Matrix
    func column(i: Int) -> Matrix {
        guard i < cols else { return Matrix(data: [[]]) } // Return an empty matrix if index is out of bounds

        var colData: [[Double]] = []
        for row in data {
            colData.append([row[i]])
        }
        return Matrix(data: colData)
    }

    // Method to multiply the matrix by a scalar
    func scale(by scalar: Double) -> Matrix {
        var scaledData = data
        for i in 0 ..< rows {
            for j in 0 ..< cols {
                scaledData[i][j] *= scalar
            }
        }
        return Matrix(data: scaledData)
    }

    // Transpose the matrix
    func transpose() -> Matrix {
        var transposedData: [[Double]] = Array(repeating: Array(repeating: 0.0, count: rows), count: cols)
        for i in 0 ..< rows {
            for j in 0 ..< cols {
                transposedData[j][i] = data[i][j]
            }
        }
        return Matrix(data: transposedData)
    }

    // Compute the inverse of the matrix
    func inverse() -> Matrix? {
        guard rows == cols else { return nil } // Must be a square matrix
        var identity = Matrix(data: Array(repeating: Array(repeating: 0.0, count: cols), count: rows))
        var copyData = data

        // Create identity matrix
        for i in 0 ..< rows {
            identity.data[i][i] = 1.0
        }

        // Gaussian elimination
        for i in 0 ..< rows {
            var maxRow = i
            for k in (i + 1) ..< rows {
                if abs(copyData[k][i]) > abs(copyData[maxRow][i]) {
                    maxRow = k
                }
            }

            // Swap rows
            let temp = copyData[maxRow]
            copyData[maxRow] = copyData[i]
            copyData[i] = temp

            let tempIdentity = identity.data[maxRow]
            identity.data[maxRow] = identity.data[i]
            identity.data[i] = tempIdentity

            let pivot = copyData[i][i]
            if pivot == 0 {
                return nil // Matrix is singular and cannot be inverted
            }

            // Scale rows
            for j in 0 ..< cols {
                copyData[i][j] /= pivot
                identity.data[i][j] /= pivot
            }

            // Eliminate other rows
            for k in 0 ..< rows {
                if k == i { continue }
                let factor = copyData[k][i]
                for j in 0 ..< cols {
                    copyData[k][j] -= factor * copyData[i][j]
                    identity.data[k][j] -= factor * identity.data[i][j]
                }
            }
        }

        return identity
    }

    func determinant() -> Double? {
        guard rows == cols else { return nil } // Must be a square matrix

        if rows == 1 {
            return data[0][0]
        }

        if rows == 2 {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0]
        }

        var det = 0.0
        for p in 0 ..< rows {
            var subMatrix = Array(repeating: Array(repeating: 0.0, count: cols - 1), count: rows - 1)
            for i in 1 ..< rows {
                var z = 0
                for j in 0 ..< cols {
                    if j == p {
                        continue
                    }
                    subMatrix[i - 1][z] = data[i][j]
                    z += 1
                }
            }
            let subMatrixObj = Matrix(data: subMatrix)
            if let subDet = subMatrixObj.determinant() {
                det += data[0][p] * subDet * (p % 2 == 0 ? 1 : -1)
            }
        }

        return det
    }

    func choleskyDecomposition() -> Matrix? {
        guard rows == cols else { return nil } // The matrix must be square

        var L = Matrix(data: Array(repeating: Array(repeating: 0.0, count: cols), count: rows))

        for i in 0 ..< rows {
            for j in 0 ... i {
                var s = 0.0
                for k in 0 ..< j {
                    s += L.data[i][k] * L.data[j][k]
                }

                if i == j {
                    let diff = data[i][i] - s
                    if diff <= 0 {
                        return nil // The matrix is not positive-definite
                    }
                    L.data[i][j] = sqrt(diff)
                } else {
                    if L.data[j][j] <= 1e-8 {
                        return nil // The matrix is not positive-definite
                    }
                    L.data[i][j] = (1.0 / L.data[j][j]) * (data[i][j] - s)
                }
            }
        }

        return L
    }

    // Method to extract the diagonal of the matrix and return it as an array of Double
    func diagonal() -> [Double] {
        guard rows == cols else { return [] }
        var diag: [Double] = []
        for i in 0 ..< rows {
            diag.append(data[i][i])
        }
        return diag
    }

    func convertToCGAffineTransform(matrix: matrix_float3x3) -> CGAffineTransform? {
        // Ensure the last row is [0, 0, 1] for a valid conversion to 2D affine transform
        guard matrix.columns.2 == simd_float3(0, 0, 1) else {
            return nil
        }

        let a = CGFloat(matrix.columns.0.x)
        let b = CGFloat(matrix.columns.0.y)
        let c = CGFloat(matrix.columns.1.x)
        let d = CGFloat(matrix.columns.1.y)
        let tx = CGFloat(matrix.columns.2.x)
        let ty = CGFloat(matrix.columns.2.y)

        return CGAffineTransform(a: a, b: b, c: c, d: d, tx: tx, ty: ty)
    }

    // Method to compute the square root of a matrix using Cholesky Decomposition
    func matrixSquareRoot() -> Matrix? {
        if let L = choleskyDecomposition() {
            return L
        }
        return nil
    }

    static func randomGaussian(mean: Matrix, covariance: Matrix) -> Matrix? {
        guard let L = covariance.choleskyDecomposition() else {
            return nil
        }

        // Generate a vector of standard normally distributed random numbers
        var z: [Double] = []
        for _ in 0 ..< mean.rows {
            z.append(Matrix.normal(Double.random(in: 0 ..< 1)))
        }

        // Convert to Matrix for easier calculations
        let zMatrix = Matrix(data: [z])

        // x = mean + L * z
        guard let Lz = L * zMatrix.transpose(),
              let x = mean + Lz
        else {
            return nil
        }

        return x
    }

    static func zeros(rows: Int, cols: Int) -> Matrix {
        let zeroRow = Array(repeating: 0.0, count: cols)
        let zeroData = Array(repeating: zeroRow, count: rows)
        return Matrix(data: zeroData)
    }

    static func normal(_ x: Double) -> Double {
        // Box-Muller method
        sqrt(-2 * log(x)) * cos(2 * Double.pi * x)
    }

    func getColumn(index: Int) -> Matrix {
        var column: [[Double]] = []
        for row in data {
            column.append([row[index]])
        }
        return Matrix(data: column)
    }

    static func += (left: inout Matrix, right: Matrix) {
        guard left.rows == right.rows, left.cols == right.cols else { return }
        for i in 0 ..< left.rows {
            for j in 0 ..< left.cols {
                left.data[i][j] += right.data[i][j]
            }
        }
    }
}

struct Orientation {
    var pitch: Double
    var roll: Double
    var yaw: Double
}

struct Quaternion {
    var x: Double
    var y: Double
    var z: Double
    var w: Double
}

// MARK: Enums

enum BlurPattern: Int, CaseIterable, Identifiable, Codable {
    case pixelated = 0
    case crystalized = 1
    case honeycombed = 2
    case gaussian = 3
    case colorblock = 4

    var id: Int {
        rawValue
    }

    var label: String {
        switch self {
        case .pixelated:
            "Pixelated"
        case .crystalized:
            "Crystalized"
        case .honeycombed:
            "Honeycombed"
        case .gaussian:
            "Gaussian"
        case .colorblock:
            "Colorblock"
        }
    }

    var description: String {
        switch self {
        case .pixelated:
            "This filter applies a pixelation effect to detected faces, creating a blocky, blurred appearance. It replaces the original facial details with a grid of larger, square pixels, effectively hiding facial features while maintaining privacy."
        case .crystalized:
            "The crystalized filter breaks down facial regions into small, crystal-like structures and blurs each one. This produces a visually interesting pattern where the original facial details transform into a blurred arrangement of crystals. This method ensures privacy while adding an artistic flair to the blurred faces."
        case .honeycombed:
            "The honeycombed filter uses a honeycomb or hexagonal pattern on detected faces, effectively distorting and hiding facial features. It replaces the original facial details with a series of interconnected hexagons, providing an aesthetically pleasing yet privacy-preserving blur."
        case .gaussian:
            "This filter applies a soft, diffused blur, obscuring facial features while keeping the overall image structure intact. This ensures privacy by preserving context and protecting identities."
        case .colorblock:
            "This filter covers faces with a solid color. While not visually appealing, this method provides maximum privacy protection by completely hiding facial features, ensuring the highest level of anonymity and full immunity to AI reconstruction algorithms."
        }
    }

    var securityRating: SecurityRating {
        switch self {
        case .pixelated:
            .medium
        case .crystalized:
            .medium
        case .honeycombed:
            .medium
        case .gaussian:
            .low
        case .colorblock:
            .high
        }
    }
}

enum SecurityRating: Int {
    case low = 1
    case medium = 2
    case high = 3
}

enum AnalyzerStatus {
    case idle
    case loading
    case analyzing
    case exporting
    case selecting
}

enum TrackType {
    case video
    case audio
}

// MARK: Classes

class Pixelate: CIFilter {
    private var filter: CIFilter

    var inputImage: CIImage?
    var scaleFactor: Int?
    var center: CIVector = .init(x: 0, y: 0)

    override init() {
        filter = CIFilter(name: "CIPixellate")!
        super.init()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func outputImage() -> CIImage? {
        guard let inputImage else { return nil }
        filter.setValue(inputImage, forKey: kCIInputImageKey)
        filter.setValue(scaleFactor, forKey: kCIInputScaleKey)
        filter.setValue(center, forKey: kCIInputCenterKey)
        return filter.outputImage?.cropped(to: inputImage.extent)
    }
}

class Crystalize: CIFilter {
    private var filter: CIFilter

    var inputImage: CIImage?
    var scaleFactor: Int?
    var center: CIVector = .init(x: 0, y: 0)

    override init() {
        filter = CIFilter(name: "CICrystallize")!
        super.init()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func outputImage() -> CIImage? {
        guard let inputImage else { return nil }
        filter.setValue(inputImage, forKey: kCIInputImageKey)
        filter.setValue(scaleFactor, forKey: kCIInputRadiusKey)
        filter.setValue(center, forKey: kCIInputCenterKey)
        return filter.outputImage?.cropped(to: inputImage.extent)
    }
}

class Hexagonal: CIFilter {
    private var filter: CIFilter

    var inputImage: CIImage?
    var scaleFactor: Int?
    var center: CIVector = .init(x: 0, y: 0)

    override init() {
        filter = CIFilter(name: "CIHexagonalPixellate")!
        super.init()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func outputImage() -> CIImage? {
        guard let inputImage else { return nil }
        filter.setValue(inputImage, forKey: kCIInputImageKey)
        filter.setValue(scaleFactor, forKey: kCIInputScaleKey)
        filter.setValue(center, forKey: kCIInputCenterKey)
        return filter.outputImage?.cropped(to: inputImage.extent)
    }
}

class Gaussian: CIFilter {
    private var filter: CIFilter

    var inputImage: CIImage?
    var scaleFactor: Int?

    override init() {
        filter = CIFilter(name: "CIGaussianBlur")!
        super.init()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func outputImage() -> CIImage? {
        guard let inputImage else { return nil }
        filter.setValue(inputImage, forKey: kCIInputImageKey)
        filter.setValue(Double(scaleFactor!) / 1.7, forKey: kCIInputRadiusKey)
        return filter.outputImage?.cropped(to: inputImage.extent)
    }
}

class Colorblock: CIFilter {
    private var filter: CIFilter

    var inputImage: CIImage?
    var inputColor: CIColor?

    override init() {
        filter = CIFilter(name: "CIConstantColorGenerator")!
        super.init()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func outputImage() -> CIImage? {
        guard let inputImage else { return nil }
        filter.setValue(inputColor, forKey: kCIInputColorKey)
        return filter.outputImage?.cropped(to: inputImage.extent)
    }
}

class FaceTrifecta: Hashable {
    let uuid = UUID()
    var faceRectangle: VNFaceObservation? // Use this for synthetic observations as well
    var faceLandmark: VNFaceObservation?
    var faceQuality: VNFaceObservation?

    var boundingBox: CGRect
    var observation: VNFaceObservation
    var confidence: VNConfidence // The confidence is published to verify if the observation is synthetic or not, since 1.0 almost always is synthetic
    var landmarks: VNFaceLandmarks?
    var quality: Float?
    var synthetic: Bool {
        confidence == 1.0
    }

    init(faceRectangle: VNFaceObservation? = nil, faceLandmark: VNFaceObservation? = nil, faceQuality: VNFaceObservation? = nil) {
        self.faceRectangle = faceRectangle
        self.faceLandmark = faceLandmark
        self.faceQuality = faceQuality

        // Since the observation will never be nil, it's safe to force unwrap these values.
        let observationBoundingBox = faceRectangle?.boundingBox ?? faceLandmark?.boundingBox ?? faceQuality?.boundingBox
        let roll = faceRectangle?.roll ?? faceLandmark?.roll ?? faceLandmark?.roll
        let yaw = faceRectangle?.yaw ?? faceLandmark?.yaw ?? faceQuality?.yaw
        let pitch = faceRectangle?.pitch ?? faceLandmark?.pitch ?? faceQuality?.pitch
        let actualConfidence = faceRectangle?.confidence ?? faceLandmark?.confidence ?? faceQuality?.confidence
        // Force unwrap is safe here as per the struct's usage contract.
        boundingBox = observationBoundingBox!
        observation = VNFaceObservation(requestRevision: 3, boundingBox: observationBoundingBox!, roll: roll!, yaw: yaw!, pitch: pitch!)
        confidence = actualConfidence!
        if faceLandmark != nil {
            landmarks = faceLandmark?.landmarks
        }
        if faceQuality != nil {
            quality = faceQuality?.faceCaptureQuality
        }
    }

    static func == (lhs: FaceTrifecta, rhs: FaceTrifecta) -> Bool {
        lhs.uuid == rhs.uuid
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(uuid)
    }
}

class SimpleChain: Hashable {
    let uuid = UUID()
    var isFinished: Bool = false // A boolean to indicate if a chain is finished (due to obscured face, facing away from camera, etc)
    var observations: [(observation: FaceTrifecta, frameNumber: Int, humanObservation: VNHumanObservation?)]
    var predictedObservations: [PredictedObservations] = []
    var kalmanState: Matrix
    var stateCovariance: Matrix
    var processNoiseCovariance: Matrix
    var measurementNoiseCovariance: Matrix
    var transitionMatrix: Matrix
    var observationMatrix: Matrix
    var frameWidth: Int
    var frameHeight: Int
    var fps: Float
    var portraitOrientation: Bool
    var rotationAngle: CGFloat
    var faceImage: URL
    var bestObservation: FaceTrifecta

    init(initialObservation: FaceTrifecta, faceImage: URL, startingFrame: Int, frameWidth: Int, frameHeight: Int, fps: Float, portraitOrientation: Bool, rotationAngle: CGFloat, humanObservation: VNHumanObservation? = nil) {
        observations = [(initialObservation, startingFrame, humanObservation)]
        bestObservation = initialObservation // The best observation is always the first observation
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.portraitOrientation = portraitOrientation
        self.fps = fps
        self.rotationAngle = rotationAngle
        self.faceImage = faceImage // The faceImage is always recorded during the first observation

        let initialMatrix = observationToMatrix(observation: initialObservation, frameWidth: frameWidth, frameHeight: frameHeight)

        kalmanState = initialMatrix

        // State Covariance
        stateCovariance = Matrix(data: [
            [100, 0, 0, 0, 0, 0, 0],
            [0, 100, 0, 0, 0, 0, 0],
            [0, 0, 100, 0, 0, 0, 0],
            [0, 0, 0, 100, 0, 0, 0],
            [0, 0, 0, 0, 100, 0, 0],
            [0, 0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 0, 100],
        ])

        // Process noise covariance
        processNoiseCovariance = Matrix(data: [
            [10, 0, 0, 0, 0, 0, 0],
            [0, 10, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0],
            [0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 0, 10, 0],
            [0, 0, 0, 0, 0, 0, 10],
        ])

        // Measurement noise covariance
        measurementNoiseCovariance = Matrix(data: [
            [5, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 5],
        ])

        transitionMatrix = Matrix(data: [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        observationMatrix = Matrix(data: [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
    }

    func addObservation(_ observation: FaceTrifecta, frame: Int, humanObservation: VNHumanObservation? = nil) {
        observations.append((observation, frame, humanObservation))

        // TODO: Check on how humanObservation can be used to improve the kalman filter via averageTrifectaPositionRelativeToHuman

        guard let predictedState = transitionMatrix * kalmanState else {
            print("Failed at predictedState")
            return
        }

        guard let predictedCovariance = predictCovariance() else {
            print("Failed at predictedCovariance")
            return
        }

        guard let innovation = calculateInnovation(predictedState: predictedState, observation: observation, frameWidth: frameWidth, frameHeight: frameHeight) else {
            print("Failed at innovation")
            return
        }

        guard let innovationCovariance = calculateInnovationCovariance(predictedCovariance: predictedCovariance) else {
            print("Failed at innovationCovariance")
            return
        }

        guard let kalmanGain = calculateKalmanGain(predictedCovariance: predictedCovariance, innovationCovariance: innovationCovariance) else {
            print("Failed at kalmanGain")
            return
        }

        guard let kalmanGainInnovation = kalmanGain * innovation else {
            print("Failed at kalmanGainInnovation")
            return
        }

        guard let updatedState = predictedState + kalmanGainInnovation else {
            print("Failed at updatedState")
            return
        }

        guard let updatedCovariance = updateCovariance(kalmanGain: kalmanGain, predictedCovariance: predictedCovariance) else {
            print("Failed at updatedCovariance")
            return
        }

        kalmanState = updatedState
        stateCovariance = updatedCovariance

        // TODO: Calculation of distance between observations should happen here

        let predictedFaceObservation = matrixToObservation(matrix: predictedState, frameWidth: frameWidth, frameHeight: frameHeight)
        let predictedObservation = PredictedObservations(observation: predictedFaceObservation, frameNumber: frame + 1)
        predictedObservations.append(predictedObservation)
    }

    func returnSelectionCard() -> SelectionCard {
        // Define a method to extract a CGImage from CIImage
        func createCGImage(imageURL: URL) -> CGImage? {
            let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil)
            let primaryIndex = CGImageSourceGetPrimaryImageIndex(source!)
            return CGImageSourceCreateImageAtIndex(source!, primaryIndex, nil)
        }

        let normalizedObservations = normalizeObservations()

        let cgImage = createCGImage(imageURL: faceImage)!
        return SelectionCard(uuid: uuid, image: cgImage, normalizedObservations: normalizedObservations)
    }

    func updateBestObservation(newFaceImage: URL, newBestObservation: FaceTrifecta) {
        if FileManager.default.fileExists(atPath: faceImage.path()) {
            try? FileManager.default.removeItem(at: faceImage)
        }
        faceImage = newFaceImage
        bestObservation = newBestObservation
    }

    private func normalizeObservations() -> [NormalizedObservation] {
        var normalizedObservations = [NormalizedObservation]()

        var lastScaleFactor = 0 // Initialize lastScaleFactor for the current chain

        for observation in observations {
            let frameNumber = observation.frameNumber
            var calcScaleFactor = 0

            let bufferWidth = observation.observation.boundingBox.width * (portraitOrientation ? 0.30 : 0.20)
            let bufferHeight = observation.observation.boundingBox.height * (portraitOrientation ? 0.20 : 0.30)
            let bufferRect = CGRectInset(observation.observation.boundingBox, -bufferWidth, -bufferHeight)
            let normalized = VNImageRectForNormalizedRect(bufferRect, frameWidth, frameHeight)
            let scaleFactor = Int(pow(normalized.width * normalized.height, 1 / 3))

            // Determine the scaleFactor based on the lastScaleFactor from the current chain
            if lastScaleFactor != 0 {
                let scaleFactorRatio = Double(scaleFactor) / Double(lastScaleFactor)
                let scaleFactorChange = abs(scaleFactorRatio - 1) * 100
                if scaleFactorChange > 15 {
                    calcScaleFactor = scaleFactor
                } else {
                    calcScaleFactor = lastScaleFactor
                }
            } else {
                calcScaleFactor = scaleFactor
            }
            lastScaleFactor = calcScaleFactor

            let normalizedObservation = NormalizedObservation(
                frameNumber: frameNumber,
                observation: normalized,
                scaleFactor: calcScaleFactor,
            )

            normalizedObservations.append(normalizedObservation)
        }

        return normalizedObservations
    }

    private func predictCovarianceMatrix(using covariance: Matrix) -> Matrix? {
        guard let temp1 = transitionMatrix * covariance,
              let temp2 = temp1 * transitionMatrix.transpose(),
              let predictedCovariance = temp2 + processNoiseCovariance
        else {
            return nil
        }
        return predictedCovariance
    }

    private func predictCovariance() -> Matrix? {
        guard let temp1 = transitionMatrix * stateCovariance,
              let temp2 = temp1 * transitionMatrix.transpose(),
              let predictedCovariance = temp2 + processNoiseCovariance
        else {
            return nil
        }
        return predictedCovariance
    }

    private func calculateInnovation(predictedState: Matrix, observation: FaceTrifecta, frameWidth: Int, frameHeight: Int) -> Matrix? {
        let observedState = observationToMatrix(observation: observation, frameWidth: frameWidth, frameHeight: frameHeight)
        guard let observationPrediction = observationMatrix * predictedState else {
            return nil
        }
        return observedState - observationPrediction
    }

    private func calculateInnovationCovariance(predictedCovariance: Matrix) -> Matrix? {
        guard let temp1 = observationMatrix * predictedCovariance,
              let temp2 = temp1 * observationMatrix.transpose(),
              let innovationCovariance = temp2 + measurementNoiseCovariance
        else {
            return nil
        }
        return innovationCovariance
    }

    private func calculateKalmanGain(predictedCovariance: Matrix, innovationCovariance: Matrix) -> Matrix? {
        guard let invInnovationCovariance = innovationCovariance.inverse(),
              let temp1 = predictedCovariance * observationMatrix.transpose(),
              let kalmanGain = temp1 * invInnovationCovariance
        else {
            return nil
        }
        return kalmanGain
    }

    private func updateCovariance(kalmanGain: Matrix, predictedCovariance: Matrix) -> Matrix? {
        guard let kalmanGainObservation = kalmanGain * observationMatrix,
              let temp1 = Matrix.identity(size: 7) - kalmanGainObservation,
              let updatedCovariance = temp1 * predictedCovariance
        else {
            return nil
        }
        return updatedCovariance
    }

    static func == (lhs: SimpleChain, rhs: SimpleChain) -> Bool {
        lhs.uuid == rhs.uuid
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(uuid)
    }
}

// MARK: Functions

func calculatePercentage(current: Double, total: Double) -> Int {
    if total == 0 { return 0 }
    if current >= total { return 100 }
    let percentage = (current / total) * 100
    return Int(percentage.rounded())
}

func generateOutputURL(inputURL: URL) -> URL {
    let path = inputURL.deletingLastPathComponent()
    let fileName = "blurred_\(inputURL.lastPathComponent)"
    let outputURL = path.appendingPathComponent(fileName)
    return outputURL
}

func analyzeFrame(imageRequestHandler: VNImageRequestHandler) -> (observations: [VNFaceObservation], humanObservations: [VNHumanObservation]) {
    var observations = [VNFaceObservation]()
    var humanObservations = [VNHumanObservation]()

    let faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: { (request: VNRequest, _: Error?) in
        if let results = request.results as? [VNFaceObservation] {
            if !results.isEmpty {
                observations.append(contentsOf: results)
            }

        } else {
            return
        }

    })

    let faceLandmarkRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request: VNRequest, _: Error?) in
        if let results = request.results as? [VNFaceObservation] {
            if !results.isEmpty {
                observations.append(contentsOf: results)
            }
        } else {
            return
        }
    })

    let faceQualityRequest = VNDetectFaceCaptureQualityRequest(completionHandler: { (request: VNRequest, _: Error?) in
        if let results = request.results as? [VNFaceObservation] {
            if !results.isEmpty {
                observations.append(contentsOf: results)
            }
        } else {
            return
        }
    })

    let humanDetectionRequest = VNDetectHumanRectanglesRequest(completionHandler: { (request: VNRequest, _: Error?) in
        if let results = request.results as? [VNHumanObservation] {
            if !results.isEmpty {
                humanObservations.append(contentsOf: results)
            }
        } else {
            return
        }
    })

    try? imageRequestHandler.perform([faceDetectionRequest, faceLandmarkRequest, faceQualityRequest, humanDetectionRequest])

    return (observations, humanObservations)
}

func observationToMatrix(observation: FaceTrifecta, frameWidth: Int, frameHeight: Int) -> Matrix {
    let boundingBox = observation.observation.boundingBox
    let zAxisValue = convertCGRectToZAxisValue(BoundingBox: boundingBox, frameWidth: frameWidth, frameHeight: frameHeight)
    let orientation = eulerToQuaternion(
        pitch: observation.observation.pitch?.doubleValue ?? 0,
        roll: observation.observation.roll?.doubleValue ?? 0,
        yaw: observation.observation.yaw?.doubleValue ?? 0,
    )

    let matrix = Matrix(data: [
        [Double(boundingBox.origin.x)],
        [Double(boundingBox.origin.y)],
        [Double(zAxisValue)],
        [Double(orientation.x)],
        [Double(orientation.y)],
        [Double(orientation.z)],
        [Double(orientation.w)],
    ])

    return matrix
}

func matrixToObservation(matrix: Matrix, frameWidth: Int, frameHeight: Int) -> FaceTrifecta {
    let dimensionValues = convertZAxisValueToCGRect(zAxisValue: matrix.data[2][0], frameWidth: frameWidth, frameHeight: frameHeight)
    let boundingBox = CGRect(
        x: CGFloat(matrix.data[0][0]),
        y: CGFloat(matrix.data[1][0]),
        width: dimensionValues.width,
        height: dimensionValues.height,
    )
    let quaternion = Quaternion(x: CGFloat(matrix.data[3][0]), y: CGFloat(matrix.data[4][0]), z: CGFloat(matrix.data[5][0]), w: CGFloat(matrix.data[6][0]))

    let orientation = quaternionToEuler(quaternion: quaternion)

    let observation = VNFaceObservation(requestRevision: 3, boundingBox: boundingBox, roll: NSNumber(value: orientation.roll), yaw: NSNumber(value: orientation.yaw), pitch: NSNumber(value: orientation.pitch))

    let trifecta = FaceTrifecta(faceRectangle: observation)

    return trifecta
}

func humanObservationToMatrix(humanObservation: VNHumanObservation, frameWidth: Int, frameHeight: Int) -> Matrix {
    let boundingBox = humanObservation.boundingBox
    let zAxisValue = convertCGRectToZAxisValue(BoundingBox: boundingBox, frameWidth: frameWidth, frameHeight: frameHeight)

    let matrix = Matrix(data: [
        [Double(boundingBox.origin.x)],
        [Double(boundingBox.origin.y)],
        [Double(zAxisValue)],
    ])

    return matrix
}

func matrixToHumanObservation(matrix: Matrix, frameWidth: Int, frameHeight: Int) -> VNHumanObservation {
    let dimensionValues = convertZAxisValueToCGRect(zAxisValue: matrix.data[2][0], frameWidth: frameWidth, frameHeight: frameHeight)
    let boundingBox = CGRect(
        x: CGFloat(matrix.data[0][0]),
        y: CGFloat(matrix.data[1][0]),
        width: dimensionValues.width,
        height: dimensionValues.height,
    )

    let observation = VNHumanObservation(boundingBox: boundingBox)

    return observation
}

func convertZAxisValueToCGRect(zAxisValue: CGFloat, frameWidth: Int, frameHeight: Int) -> (width: CGFloat, height: CGFloat) {
    let frameArea = CGFloat(frameWidth * frameHeight)
    let zAxisZeroPoint = frameArea / 4.0

    let adjustedFaceArea = zAxisValue + zAxisZeroPoint

    let ratio = adjustedFaceArea / frameArea
    let adjustedWidth = sqrt(ratio * CGFloat(frameHeight * frameHeight))
    let adjustedHeight = sqrt(ratio * CGFloat(frameWidth * frameWidth))

    return (adjustedWidth, adjustedHeight)
}

func convertCGRectToZAxisValue(BoundingBox: CGRect, frameWidth: Int, frameHeight: Int) -> CGFloat {
    let frameArea = frameWidth * frameHeight
    let zAxisZeroPoint = frameArea / 4

    let faceArea = BoundingBox.width * BoundingBox.height
    let zAxisValue = faceArea - CGFloat(zAxisZeroPoint)

    return zAxisValue
}

func eulerToQuaternion(pitch: Double, roll: Double, yaw: Double) -> Quaternion {
    let cy = cos(yaw * 0.5)
    let sy = sin(yaw * 0.5)
    let cp = cos(pitch * 0.5)
    let sp = sin(pitch * 0.5)
    let cr = cos(roll * 0.5)
    let sr = sin(roll * 0.5)

    let w = cr * cp * cy + sr * sp * sy
    let x = sr * cp * cy - cr * sp * sy
    let y = cr * sp * cy + sr * cp * sy
    let z = cr * cp * sy - sr * sp * cy

    return Quaternion(x: x, y: y, z: z, w: w)
}

func quaternionToEuler(quaternion: Quaternion) -> Orientation {
    let x = quaternion.x
    let y = quaternion.y
    let z = quaternion.z
    let w = quaternion.w

    let sinr_cosp = 2 * (w * x + y * z)
    let cosr_cosp = 1 - 2 * (x * x + y * y)
    let roll = atan2(sinr_cosp, cosr_cosp)

    let sinp = 2 * (w * y - z * x)
    let pitch: Double = if abs(sinp) >= 1 {
        copysign(Double.pi / 2, sinp)
    } else {
        asin(sinp)
    }

    let siny_cosp = 2 * (w * z + x * y)
    let cosy_cosp = 1 - 2 * (y * y + z * z)
    let yaw = atan2(siny_cosp, cosy_cosp)

    return Orientation(pitch: pitch, roll: roll, yaw: yaw)
}

func calculateFaceObservationSimilarity(face1: VNFaceObservation, face2: VNFaceObservation) -> Float {
    let boundingBox1 = face1.boundingBox
    let boundingBox2 = face2.boundingBox
    let overlapSimilarity = calculateBoundingBoxOverlapPercentage(boundingBox1: boundingBox1, boundingBox2: boundingBox2)
    let positionSimilarity = calculateBoundingBoxPositionPercentage(boundingBox1: boundingBox1, boundingBox2: boundingBox2)

    let orientationSimilarity = calculateHeadPoseSimilarityPercentage(face1: face1, face2: face2)

    let weightedOverlapSimilarity = overlapSimilarity * 0.4
    let weightedPositionSimilarity = positionSimilarity * 0.4
    let weightedOrientationSimilarity = orientationSimilarity * 0.2

    return weightedOverlapSimilarity + weightedPositionSimilarity + weightedOrientationSimilarity
}

func calculateBoundingBoxPositionPercentage(boundingBox1: CGRect, boundingBox2: CGRect) -> Float {
    let center1 = CGPoint(x: boundingBox1.midX, y: boundingBox1.midY)
    let center2 = CGPoint(x: boundingBox2.midX, y: boundingBox2.midY)
    let bboxDistance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2))
    let bboxSimilarity = max(0, 1.0 - Float(bboxDistance))

    return bboxSimilarity
}

func calculateBoundingBoxOverlapPercentage(boundingBox1: CGRect, boundingBox2: CGRect) -> Float {
    let intersection = boundingBox1.intersection(boundingBox2)
    let union = boundingBox1.union(boundingBox2)
    let overlapPercentage = Float(intersection.width * intersection.height) / Float(union.width * union.height)
    return overlapPercentage
}

func calculateHeadPoseSimilarityPercentage(face1: VNFaceObservation, face2: VNFaceObservation) -> Float {
    guard let yaw1 = face1.yaw, let yaw2 = face2.yaw, let pitch1 = face1.pitch, let pitch2 = face2.pitch, let roll1 = face1.roll, let roll2 = face2.roll else {
        return 0.0
    }
    let yawAngleDifference = abs(Float(truncating: yaw1) - Float(truncating: yaw2))
    let pitchAngleDifference = abs(Float(truncating: pitch1) - Float(truncating: pitch2))
    let rollAngleDifference = abs(Float(truncating: roll1) - Float(truncating: roll2))
    let angleDifferenceSum = yawAngleDifference + pitchAngleDifference + rollAngleDifference
    let headPoseSimilarityPercentage = 1.0 - (angleDifferenceSum / 3.0) // Normalize to 0-1 range
    return headPoseSimilarityPercentage
}

func collectFaceObservations(rawObservations: [VNFaceObservation]) -> [FaceTrifecta] {
    guard rawObservations.count > 1 else {
        return rawObservations.map { FaceTrifecta(faceRectangle: $0) }
    }

    var faceTrifectas = [FaceTrifecta]()
    var processedObservations = rawObservations

    while !processedObservations.isEmpty {
        let currentObservation = processedObservations.removeFirst()
        var faceRectangle: VNFaceObservation? = currentObservation.landmarks != nil ? currentObservation : nil
        var faceLandmark: VNFaceObservation? = currentObservation.landmarks != nil ? currentObservation : nil
        var faceQuality: VNFaceObservation? = currentObservation.faceCaptureQuality != nil ? currentObservation : nil

        for (index, otherObservation) in processedObservations.enumerated().reversed() {
            if approximatelyEqual(from: currentObservation, to: otherObservation) {
                if faceRectangle == nil, otherObservation.landmarks == nil {
                    faceRectangle = otherObservation
                }
                if faceLandmark == nil, otherObservation.landmarks != nil {
                    faceLandmark = otherObservation
                }
                if faceQuality == nil, otherObservation.faceCaptureQuality != nil {
                    faceQuality = otherObservation
                }
                processedObservations.remove(at: index)
            }
        }

        let trifecta = FaceTrifecta(faceRectangle: faceRectangle, faceLandmark: faceLandmark, faceQuality: faceQuality)
        faceTrifectas.append(trifecta)
    }

    return faceTrifectas
}

func approximatelyEqual(from obs1: VNFaceObservation, to obs2: VNFaceObservation, epsilon: CGFloat = 0.01) -> Bool {
    let b1 = obs1.boundingBox
    let b2 = obs2.boundingBox

    let boxApproxEqual = abs(b1.origin.x - b2.origin.x) <= epsilon &&
        abs(b1.origin.y - b2.origin.y) <= epsilon &&
        abs(b1.size.width - b2.size.width) <= epsilon &&
        abs(b1.size.height - b2.size.height) <= epsilon

    let pitchApproxEqual = abs((obs1.pitch?.doubleValue ?? 0) - (obs2.pitch?.doubleValue ?? 0)) <= Double(epsilon)
    let rollApproxEqual = abs((obs1.roll?.doubleValue ?? 0) - (obs2.roll?.doubleValue ?? 0)) <= Double(epsilon)
    let yawApproxEqual = abs((obs1.yaw?.doubleValue ?? 0) - (obs2.yaw?.doubleValue ?? 0)) <= Double(epsilon)

    return boxApproxEqual && pitchApproxEqual && rollApproxEqual && yawApproxEqual
}

func colorFromHex(hex: String, alpha: CGFloat = 1.0) -> UIColor {
    var hexValue = hex.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()

    if hexValue.hasPrefix("#") {
        hexValue.remove(at: hexValue.startIndex)
    }

    var rgbValue: UInt64 = 0
    Scanner(string: hexValue).scanHexInt64(&rgbValue)

    let red = CGFloat((rgbValue & 0xFF0000) >> 16) / 255.0
    let green = CGFloat((rgbValue & 0x00FF00) >> 8) / 255.0
    let blue = CGFloat(rgbValue & 0x0000FF) / 255.0

    return UIColor(red: red, green: green, blue: blue, alpha: alpha)
}

func matchTrifectaToHuman(
    faceTrifectas: [FaceTrifecta],
    humanObservations: [VNHumanObservation],
) -> (
    trifectaHumanPairs: [(FaceTrifecta, VNHumanObservation?)],
    unmatchedHumans: Set<VNHumanObservation>
) {
    var trifectaHumanPairs: [(FaceTrifecta, VNHumanObservation?)] = []
    var unmatchedTrifectas: Set<FaceTrifecta> = Set(faceTrifectas)
    var unmatchedHumans: Set<VNHumanObservation> = Set(humanObservations)

    for trifecta in faceTrifectas {
        for human in humanObservations {
            if isTrifectaInsideHuman(trifecta: trifecta, human: human),
               isTrifectaSizeRelevantComparedToHuman(trifecta: trifecta, human: human)
            {
                let closestFace = findClosestTrifectaToHuman(faceTrifectas: [trifecta], humanObservation: human)

                if closestFace === trifecta {
                    trifectaHumanPairs.append((trifecta, human))
                    unmatchedTrifectas.remove(trifecta)
                    unmatchedHumans.remove(human)
                    break
                }
            }
        }
    }

    // Append unmatched trifectas with nil VNHumanObservation
    for trifecta in unmatchedTrifectas {
        trifectaHumanPairs.append((trifecta, nil))
    }

    return (trifectaHumanPairs, unmatchedHumans)
}

func isTrifectaInsideHuman(trifecta: FaceTrifecta, human: VNHumanObservation) -> Bool {
    trifecta.boundingBox.minX >= human.boundingBox.minX &&
        trifecta.boundingBox.maxX <= human.boundingBox.maxX &&
        trifecta.boundingBox.minY >= human.boundingBox.minY &&
        trifecta.boundingBox.maxY <= human.boundingBox.maxY
}

func isTrifectaSizeRelevantComparedToHuman(trifecta: FaceTrifecta, human: VNHumanObservation) -> Bool {
    let faceArea = trifecta.boundingBox.width * trifecta.boundingBox.height
    let humanArea = human.boundingBox.width * human.boundingBox.height

    let ratio = faceArea / humanArea

    let lowerThreshold: CGFloat = 0.05
    let upperThreshold: CGFloat = 0.35

    return ratio > lowerThreshold && ratio < upperThreshold
}

func findClosestTrifectaToHuman(faceTrifectas: [FaceTrifecta], humanObservation: VNHumanObservation) -> FaceTrifecta? {
    var closestFace: FaceTrifecta?
    var minDistance = Double.infinity

    let humanCenter = CGPoint(x: humanObservation.boundingBox.midX, y: humanObservation.boundingBox.midY)

    for trifecta in faceTrifectas {
        let faceCenter = CGPoint(x: trifecta.boundingBox.midX, y: trifecta.boundingBox.midY)
        let dx = humanCenter.x - faceCenter.x
        let dy = humanCenter.y - faceCenter.y
        let distance = dx * dx + dy * dy

        if distance < minDistance {
            closestFace = trifecta
            minDistance = distance
        }
    }

    return closestFace
}

func findClosestHumanToFace(humanObservations: [VNHumanObservation], faceTrifecta: FaceTrifecta) -> VNHumanObservation? {
    var closestHuman: VNHumanObservation?
    var minDistance = Double.infinity

    let faceCenter = CGPoint(x: faceTrifecta.boundingBox.midX, y: faceTrifecta.boundingBox.midY)

    for human in humanObservations {
        let humanCenter = CGPoint(x: human.boundingBox.midX, y: human.boundingBox.midY)
        let dx = humanCenter.x - faceCenter.x
        let dy = humanCenter.y - faceCenter.y
        let distance = dx * dx + dy * dy

        if distance < minDistance {
            closestHuman = human
            minDistance = distance
        }
    }

    return closestHuman
}

func clearCache() {
    let fileManager = FileManager.default
    do {
        let documentDirectoryURL = AppEnvironment.shared.storageDirectory

        let fileURLs = try fileManager.contentsOfDirectory(at: documentDirectoryURL, includingPropertiesForKeys: nil, options: .skipsHiddenFiles) // Enables hidden directories to still contain data
        for url in fileURLs {
            try fileManager.removeItem(at: url)
        }
    } catch {
        print("Clearning of cache failed with error: \(error)")
    }
}
