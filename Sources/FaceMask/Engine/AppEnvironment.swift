
import Foundation

struct AppEnvironment {
    let storageDirectory: URL

    static let shared: AppEnvironment = {
        guard let environment = AppEnvironment() else {
            fatalError("Failed to initialize AppEnvironment. Ensure all required environment variables are set.")
        }
        return environment
    }()

    private init?() {
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
    }
}
