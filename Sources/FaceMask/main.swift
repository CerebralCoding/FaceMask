import AVKit
import Foundation

typealias UIColor = NSColor

func main() {
    let env = AppEnvironment.shared

    Task {
        let progress = Progress(totalUnitCount: 200)

        ProgressBar.start(progress: progress)

        do {
            CommandLine.info([
                "Starting blur of file: \(env.inputURL.lastPathComponent) ",
                "with blur pattern: \(env.userConfig.blurPattern) ",
                "circular shape: \(env.userConfig.circularShape) ",
            ].joined())

            let videoAnalyzer = VideoAnalyzer()

            videoAnalyzer.onAnalyzedPercentageUpdate = { percentage in
                progress.completedUnitCount = Int64(percentage)
            }

            videoAnalyzer.onWrittenPercentageUpdate = { percentage in
                progress.completedUnitCount = 100 + Int64(percentage)
            }

            guard let outputURL = try await videoAnalyzer.analyze(inputURL: env.inputURL, userConfig: env.userConfig, selectiveBlur: env.selectiveBlur) else {
                return
            }

            ProgressBar.stop()
            CommandLine.success("Success! File saved at: \(outputURL.path())")
            clearCache()
            exit(0)
        } catch let error as AnalyzerError {
            ProgressBar.stop()
            CommandLine.error("Failed to process video: \(error.debugDescription)")
            clearCache()
            exit(1)
        }
    }

    RunLoop.main.run()
}

main()
