import AVKit
import Foundation

typealias UIColor = NSColor

func main() {
    let allCasesString = BlurPattern.allCases.map { "\($0.rawValue)" }.joined(separator: ", ")

    let commandlineArguments = ProcessInfo.processInfo.arguments

    let selectiveBlur = false

    guard commandlineArguments.count >= 2 else {
        print("Usage: facemask <path_to_video> [<blur_pattern> <circularShape? true/false> <hexColor>]")
        print("Valid blur patterns: \(allCasesString)")
        exit(1)
    }

    let inputURL = URL(fileURLWithPath: commandlineArguments[1])

    let blurPattern: BlurPattern = if commandlineArguments.count > 2, let faceBlur = Int(commandlineArguments[2]), let variable = BlurPattern(rawValue: faceBlur) {
        variable
    } else {
        BlurPattern(rawValue: 0)! // Assuming 0 is a valid rawValue for a default BlurPattern
    }

    let circularShape: Bool = if commandlineArguments.count > 3, let circularValue = Bool(commandlineArguments[3]) {
        circularValue
    } else {
        true
    }

    let hexColor: String = if commandlineArguments.count > 4 {
        commandlineArguments[4]
    } else {
        "#000000"
    }

    let userConfig = UserConfig(blurPattern: blurPattern, circularShape: circularShape, hexColor: hexColor)

    let outputURL = generateOutputURL(inputURL: inputURL)

    do {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }
    } catch {
        print("Couldn't remove file")
        exit(1)
    }
    Task {
        do {
            let videoAnalyzer = VideoAnalyzer()

            videoAnalyzer.onStatusUpdate = { status in
                print(status)
            }

            videoAnalyzer.onAnalyzedPercentageUpdate = { analyzedPercentage in
                print("Analyzed: \(analyzedPercentage)%")
            }

            videoAnalyzer.onWrittenPercentageUpdate = { writtenPercentage in
                print("Exported: \(writtenPercentage)%")
            }

            guard let outputURL = try? await videoAnalyzer.analyze(inputURL: inputURL, userConfig: userConfig, selectiveBlur: selectiveBlur) else {
                throw AnalyzerError("Failed to process video")
            }

            print("Success! File saved at: \(outputURL.path())")
            clearCache()
            exit(0)
        } catch let error as AnalyzerError {
            print(error.debugDescription)
            clearCache()
            exit(1)
        }
    }

    RunLoop.main.run()
}

main()
