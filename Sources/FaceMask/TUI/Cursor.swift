import Foundation

public enum Cursor {
    public static func moveUp(by rows: Int = 1) {
        print("\(CSI)\(rows)A", terminator: "")
    }

    public static func move(toRow row: Int, col: Int) {
        print("\(CSI)\(row);\(col)H", terminator: "")
    }

    public static func hide() {
        print("\(CSI)?25l", terminator: "")
    }

    public static func show() {
        print("\(CSI)?25h", terminator: "")
    }

    public static func clearBelow() {
        print("\(CSI)0J", terminator: "")
    }

    public static func clearToEndOfLine() {
        print("\(CSI)0K", terminator: "")
    }
}
