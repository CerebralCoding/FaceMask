import Foundation

public enum Key {
    public static let up = "\u{1B}[A"
    public static let down = "\u{1B}[B"

    public static func read() -> String {
        String(decoding: FileHandle.standardInput.availableData, as: UTF8.self)
    }
}
