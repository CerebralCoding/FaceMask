
import Foundation

struct AppEnvironment {
    let storageDirectory: URL
    let inputURL: URL
    let outputURL: URL
    let userConfig: UserConfig
    let selectiveBlur: Bool

    static let shared: AppEnvironment = {
        AppEnvironment()
    }()

    private init() {
        func ensureDirectoryExists(at path: URL, label: String) {
            if !FileManager.default.fileExists(atPath: path.path) {
                do {
                    try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true, attributes: nil)
                    print("Created \(label) directory at \(path.path)")
                } catch {
                    fatalError("Failed to create \(label) directory: \(error)")
                }
            }
        }

        let homeDirectory = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".config")
        let storageDirectory = homeDirectory.appendingPathComponent("FaceMask", isDirectory: true)
        ensureDirectoryExists(at: storageDirectory, label: "storage")

        self.storageDirectory = storageDirectory

        let allCasesString = BlurPattern.allCases.map { "\($0.rawValue)" }.joined(separator: ", ")
        let commandlineArguments = ProcessInfo.processInfo.arguments
        guard commandlineArguments.count >= 2 else {
            CommandLine.error("No arguments provided.")
            CommandLine.warn("Usage: facemask <path_to_video> [<blur_pattern> <circularShape? true/false> <hexColor>]")
            CommandLine.warn("Valid blur patterns: \(allCasesString)")
            exit(1)
        }

        let inputURL = URL(fileURLWithPath: commandlineArguments[1])

        let blurPattern: BlurPattern = if commandlineArguments.count > 2,
            let faceBlur = Int(commandlineArguments[2]),
            let variable = BlurPattern(rawValue: faceBlur) {
            variable
        } else {
            BlurPattern(rawValue: 0)! // Assuming 0 is a valid rawValue for a default BlurPattern
        }

        let circularShape: Bool = if commandlineArguments.count > 3,
            let circularValue = Bool(commandlineArguments[3]) {
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
            CommandLine.error("Couldn't remove file")
            exit(1)
        }

        self.inputURL = inputURL
        self.outputURL = outputURL
        self.userConfig = userConfig
        self.selectiveBlur = false
    }
}
