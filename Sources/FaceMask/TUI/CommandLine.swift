import Foundation

public enum CommandLine {
    public static func chooseFromList<T: CustomStringConvertible>(_ list: [T], prompt: String) -> T {
        assert(!list.isEmpty)

        if list.count == 1 {
            print([
                "?".brightGreen.bold,
                prompt.bold,
                list[0].description.brightCyan,
            ].joined(separator: " "))
            return list[0]
        }

        Termios.enterRawMode()
        Cursor.hide()
        defer {
            Cursor.show()
            Termios.pop()
        }

        var selectedIndex = 0
        var filter = ""
        let windowRows = Window.size?.rows ?? 10
        while true {
            print([
                "?".brightGreen.bold,
                prompt.bold,
                filter,
                "[Use arrows to move, type to filter]".brightCyan,
            ].filter { !$0.isEmpty }.joined(separator: " "))

            var filteredList = list
            if !filter.isEmpty {
                filteredList = list.filter {
                    $0.description.lowercased().contains(filter.lowercased())
                }
            }
            filteredList = Array(filteredList.prefix(windowRows - 2))
            selectedIndex = min(selectedIndex, filteredList.count - 1)
            if filteredList.isEmpty {
                selectedIndex = -1
            } else if selectedIndex < 0 {
                selectedIndex = 0
            }

            for (index, option) in filteredList.enumerated() {
                if index == selectedIndex {
                    print("> \(option)".removingSGR().brightCyan.bold)
                } else {
                    print("  \(option)")
                }
            }
            let key = Key.read()
            switch key {
            case Key.up:
                selectedIndex = max(selectedIndex - 1, 0)
            case Key.down:
                selectedIndex = min(selectedIndex + 1, filteredList.count - 1)
            case "\n":
                guard filteredList.indices.contains(selectedIndex) else { break }
                Cursor.moveUp(by: filteredList.count + 1)
                Cursor.clearBelow()
                print([
                    "?".brightGreen.bold,
                    prompt.bold,
                    filteredList[selectedIndex].description.brightCyan,
                ].joined(separator: " "))
                return filteredList[selectedIndex]
            case "\u{7F}":
                if !filter.isEmpty { filter.removeLast() }
            default:
                if Character(key).isASCII {
                    filter += key
                }
            }
            Cursor.moveUp(by: filteredList.count + 1)
            Cursor.clearBelow()
        }
    }

    public static func askForText(prompt: String = "?", title: String? = nil) -> String {
        if let title { print(title.brightBlue.bold) }
        print(prompt.brightMagenta, terminator: " ")
        return readLine() ?? ""
    }

    public static func info(_ message: String) {
        print("i ".brightCyan.bold + message)
    }

    public static func warn(_ message: String) {
        print("⚠ ".brightYellow.bold + message)
    }

    public static func error(_ message: String) {
        print("× ".brightRed.bold + message)
    }

    public static func success(_ message: String) {
        print("✓ ".brightGreen.bold + message)
    }
}
