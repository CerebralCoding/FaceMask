import Foundation

public enum ProgressBar {
    // MARK: Public

    public static func start(progress: Progress) {
        task = Task {
            let frames = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▊", "▉"]
            while !Task.isCancelled {
                let cols = (Window.size?.cols ?? 10) - 2 - 8
                let completed = Int(floor(progress.fractionCompleted * Double(cols)))
                let partial = (progress.fractionCompleted * Double(cols)).truncatingRemainder(dividingBy: 1)
                var components = ["[".brightCyan.bold]
                components.append(contentsOf: Array(repeating: "█", count: completed))
                if cols - completed - 1 > 0 {
                    components.append(frames[Int(floor(partial * 8))])
                    components.append(contentsOf: Array(repeating: " ", count: cols - completed - 1))
                }
                components.append("]".brightCyan.bold)
                components.append(String(format: " %.2f%%", progress.fractionCompleted * 100).gray)
                print(components.joined())
                Cursor.moveUp()
                try? await Task.sleep(nanoseconds: 80_000_000)
                guard !Task.isCancelled else { return }
                Cursor.clearToEndOfLine()
            }
        }
    }

    public static func stop() {
        task?.cancel()
        Cursor.clearToEndOfLine()
    }

    // MARK: Private

    private nonisolated(unsafe) static var task: Task<Void, Never>?
}
